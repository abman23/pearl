/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Manages Wi-Fi Aware connections.
*/

import Foundation
import WiFiAware
import Network
import OSLog

typealias WiFiAwareConnection = NetworkConnection<Coder<NetworkEvent, NetworkEvent, NetworkJSONCoder>>
typealias WiFiAwareConnectionID = String
typealias WiFiAwareConnectionState = (WiFiAwareConnection, WiFiAwareConnection.State)

private struct ConnectionInfo {
    let receiverTask: Task<Void, Error>
    let stateUpdateTask: Task<Void, Error>
    var remoteDevice: WAPairedDevice?
}

actor ConnectionManager: Sendable {
    private var connections: [WiFiAwareConnectionID: WiFiAwareConnection] = [:]
    private var connectionsInfo: [WiFiAwareConnectionID: ConnectionInfo] = [:]

    public let localEvents: AsyncStream<LocalEvent>
    private let localEventsContinuation: AsyncStream<LocalEvent>.Continuation

    public let networkEvents: AsyncStream<NetworkEvent>
    private let networkEventsContinuation: AsyncStream<NetworkEvent>.Continuation
    
    private var pingSequenceNumber = 0
    private var pendingPings: [Int: Date] = [:]

    init() async {
        (self.localEvents, self.localEventsContinuation) = AsyncStream.makeStream(of: LocalEvent.self)
        (self.networkEvents, self.networkEventsContinuation) = AsyncStream.makeStream(of: NetworkEvent.self)
    }

// MARK: - Setup

    func add(_ connection: WiFiAwareConnection) {
        logger.info("Add connection: \(connection.debugDescription)")

        connectionsInfo[connection.id] = .init(receiverTask: setupReceiver(connection),
                                              stateUpdateTask: setupStateUpdateHandler(connection))
    }

// MARK: - State Updates

    private func setupStateUpdateHandler(_ connection: WiFiAwareConnection) -> Task<Void, Error> {
        let (stream, continuation) = AsyncStream.makeStream(of: WiFiAwareConnectionState.self)

        connection.onStateUpdate { connection, state in
            logger.info("\(connection.debugDescription): \(String(describing: state))")
            continuation.yield((connection, state))
        }

        return Task {
            for await (connection, state) in stream {
                switch state {
                case .setup, .waiting, .preparing: break

                case .ready:
                    connections[connection.id] = connection

                    if let wifiAwarePath = try await connection.currentPath?.wifiAware {
                        let connectedDevice = wifiAwarePath.endpoint.device
                        let performanceReport = wifiAwarePath.performance

                        let detail = ConnectionDetail(connection: connection, performanceReport: performanceReport)
                        localEventsContinuation.yield(.connectionReady(connectedDevice, detail))

                        connectionsInfo[connection.id]?.remoteDevice = connectedDevice
                    }

                case .failed, .cancelled:
                    if state != .cancelled {
                        stop(connection)
                    }
                    guard let disconnectedDevice = connectionsInfo[connection.id]?.remoteDevice else { continue }
                    localEventsContinuation.yield(.connectionStopped(disconnectedDevice, connection.id))
                @unknown default: break
                }
            }
        }
    }

// MARK: - Receive

    private func setupReceiver(_ connection: WiFiAwareConnection) -> Task<Void, Error> {
        logger.info("Set up receiver: \(connection.debugDescription)")

        return Task {
            for try await (event, _) in connection.messages {
                switch event {
                    case .ping(let sequenceNumber):
                        // Immediately respond with pong
                        let pongEvent = NetworkEvent.pong(sequenceNumber: sequenceNumber)
                        await send(pongEvent, to: connection)
                        
                    case .pong(let sequenceNumber):
                        pendingPings[sequenceNumber] = Date()
                    default:
                        networkEventsContinuation.yield(event)
                }
            }
        }
    }

// MARK: - Send

    func send(_ event: NetworkEvent, to connection: WiFiAwareConnection) async {
        do {
            try await connection.send(event)
        } catch {
            logger.error("Failed to send to: \(connection.debugDescription): \(error)")
        }
    }

    func sendToAll(_ event: NetworkEvent) async {
        for connection in connections.values {
            await send(event, to: connection)
        }
    }

// MARK: - Chunked File Transfer
    
    func sendFileInChunks(_ data: Data, filename: String, chunkSize: Int, to connection: WiFiAwareConnection) async {
        let receiverDevice: WAPairedDevice? = try? await connection.currentPath?.wifiAware?.endpoint.device
        let receiver = getDeviceInfo(device: receiverDevice)
        
        let totalChunks = (data.count + chunkSize - 1) / chunkSize
        
        logger.info("Starting chunked file transfer: \(filename), chunks: \(totalChunks)")
        
        // Send start transfer notification
        await send(.startFileTransfer(filename: filename, chunkSize: chunkSize, totalChunks: totalChunks, battery: getBatteryInfo(), receiver: receiver), to: connection)
        
        // Send file in chunks
        for chunkIndex in 0..<totalChunks {
            logger.info("Transmitting chunk \(chunkIndex+1)/\(totalChunks)")
            let startIndex = chunkIndex * chunkSize
            let endIndex = min(startIndex + chunkSize, data.count)
            let chunkData = data[startIndex..<endIndex]
            
            let chunkEvent = NetworkEvent.fileChunk(data: Data(chunkData), chunkIndex: chunkIndex, filename: filename, timestamp: Date())
            await send(chunkEvent, to: connection)
            
//            // Small delay to prevent overwhelming the network
//            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms delay
        }
        
        // Send end transfer notification
        let senderLatency: Double = await (measureRTT(to: connection) ?? -2.0) / 2.0
        await send(.endFileTransfer(filename: filename, latency: senderLatency), to: connection)
        logger.info("Completed chunked file transfer: \(filename)")
    }
    
    func sendFileToAll(_ data: Data, filename: String) async {
        for connection in connections.values {
            // Reserve space for protocol overhead (headers, JSON encoding, etc.)
            // Use a conservative chunk size to avoid buffer overflow
            let maxDatagramSize = connection.maximumDatagramSize
            let chunkSize = max(1024, min(maxDatagramSize - 512, 1400)) // Conservative approach
            
            logger.info("File size: \(data.count), max datagram: \(maxDatagramSize), using chunk size: \(chunkSize)")
            await sendFileInChunks(data, filename: filename, chunkSize: chunkSize, to: connection)
        }
    }
    
// MARK: - Monitor

    func monitor() async throws {
        for connection in connections.values.filter({ $0.state == .ready }) {
            if let wifiAwarePath = try await connection.currentPath?.wifiAware {
                let connectedDevice = wifiAwarePath.endpoint.device
                let performanceReport = wifiAwarePath.performance

                let performance = ConnectionDetail(connection: connection, performanceReport: performanceReport)
                localEventsContinuation.yield(.connectionPerformanceUpdate(connectedDevice, performance))
            }
        }
    }
    
    func measureRTT(to connection: WiFiAwareConnection) async -> Double? {
        var rtts: [Double] = []
        for _ in 0..<10 {
            let sequenceNumber = pingSequenceNumber
            pingSequenceNumber += 1
            
            let sendTime = Date()
            // Send ping
            await send(.ping(sequenceNumber: sequenceNumber), to: connection)
            
            // Wait for pong with timeout
            let timeout = 5.0 // 5 seconds timeout
            let startWait = Date()
            
            while Date().timeIntervalSince(startWait) < timeout {
                if let receiveTime = pendingPings[sequenceNumber] {
                    // Pong received, calculate RTT
                    let rtt = receiveTime.timeIntervalSince(sendTime) * 1000 // Convert to milliseconds
//                    logger.info("RTT measurement: \(String(format: "%.2f", rtt))ms")
                    pendingPings.removeValue(forKey: sequenceNumber)
                    rtts.append(rtt)
                    break
                }
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms check interval
            }
        }
        
        return median(of: rtts)
    }

// MARK: - Teardown

    func stop(_ connection: WiFiAwareConnection) {
        logger.info("Stop connection: \(connection.debugDescription)")
        connectionsInfo[connection.id]?.receiverTask.cancel()
        if let removedConnection = connections.removeValue(forKey: connection.id) {
            logger.info("Removed: \(removedConnection.debugDescription)")
        }
    }

    func invalidate(_ id: WiFiAwareConnectionID) {
        logger.info("Invalidate connection ID: \(id)")
        connectionsInfo[id]?.stateUpdateTask.cancel()
        connectionsInfo.removeValue(forKey: id)
    }

    deinit {
        for info in connectionsInfo.values {
            info.receiverTask.cancel()
            info.stateUpdateTask.cancel()
        }
        connections.removeAll()

        localEventsContinuation.finish()
        networkEventsContinuation.finish()
    }
}
