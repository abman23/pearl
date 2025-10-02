/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Manages Wi-Fi Aware networking.
*/

import WiFiAware
import Network
import OSLog

actor NetworkManager {
    public let localEvents: AsyncStream<LocalEvent>
    private let localEventsContinuation: AsyncStream<LocalEvent>.Continuation

    public let networkEvents: AsyncStream<NetworkEvent>
    private let networkEventsContinuation: AsyncStream<NetworkEvent>.Continuation

    private let connectionManager: ConnectionManager
    
    // Current configuration
    var currentPerformanceMode: WAPerformanceMode = appPerformanceMode
    var currentAccessCategory: WAAccessCategory = appAccessCategory
    var configureChangePending: Bool = false

    init(connectionManager: ConnectionManager) {
        (self.localEvents, self.localEventsContinuation) = AsyncStream.makeStream(of: LocalEvent.self)
        (self.networkEvents, self.networkEventsContinuation) = AsyncStream.makeStream(of: NetworkEvent.self)

        self.connectionManager = connectionManager
    }
    
    func setupConnection(to endpoint: WAEndpoint) async {
        let connection = NetworkConnection(
            to:
                endpoint,
            using: .parameters {
                Coder(receiving: NetworkEvent.self, sending: NetworkEvent.self, using: NetworkJSONCoder()) {
//                    UDP()
                    TCP()
                }
            }
            .wifiAware { $0.performanceMode = currentPerformanceMode }
                .serviceClass(currentAccessCategory.serviceClass)
        )

        logger.info("Set up connection: \(connection.debugDescription)\nto: \(endpoint)")
        logger.info("Connection max datagram size: \(connection.maximumDatagramSize)")
        
        await self.connectionManager.add(connection)
    }
    
// MARK: - Configuration Management
    
    func changeConfiguration(performanceMode: WAPerformanceMode, accessCategory: WAAccessCategory, mode: SimulationEngine.Mode) async -> Bool {
//        logger.info("Changing configuration to: \(performanceMode.descirption), \(accessCategory.description)")
        if performanceMode == currentPerformanceMode && accessCategory == currentAccessCategory {
            return false
        }
        // Store new configuration
        currentPerformanceMode = performanceMode
        currentAccessCategory = accessCategory
        
        return true
    }

// MARK: - NetworkListener (Publisher)

    func listen() async throws {
        logger.info("Start NetworkListener with config: \(self.currentPerformanceMode.descirption), \(self.currentAccessCategory.description)")
        
        // Check paired devices first
        do {
            if let devices = try await WAPairedDevice.allDevices.current() {
                logger.info("Found \(devices.count) paired devices:")
                for (id, device) in devices {
                    logger.info("  - \(device.displayName) (ID: \(id))")
                }
            } else {
                logger.warning("No paired devices found")
            }
        } catch {
            logger.error("Failed to get paired devices: \(error)")
        }

        try await NetworkListener(for:
            .wifiAware(.connecting(to: .simulationService, from: .allPairedDevices)),
        using: .parameters {
            Coder(receiving: NetworkEvent.self, sending: NetworkEvent.self, using: NetworkJSONCoder()) {
//                UDP()
                TCP()
            }
        }
        .wifiAware { $0.performanceMode = currentPerformanceMode }
            .serviceClass(currentAccessCategory.serviceClass))
        .onStateUpdate { listener, state in
            logger.info("\(String(describing: listener)): \(String(describing: state))")

            switch state {
            case .setup, .waiting: break
            case .ready:
                self.localEventsContinuation.yield(.listenerRunning)
            case .failed, .cancelled: self.localEventsContinuation.yield(.listenerStopped)
            default: break
            }
        }
        .run { connection in
            logger.info("Received connection: \(String(describing: connection))")
            await self.connectionManager.add(connection)
        }
    }

// MARK: - NetworkBrowser (Subscriber)

    func browse() async throws {
        logger.info("Start NetworkBrowser")
        
        // Check paired devices first
        do {
            if let devices = try await WAPairedDevice.allDevices.current() {
                logger.info("Found \(devices.count) paired devices:")
                for (id, device) in devices {
                    logger.info("  - \(device.displayName) (ID: \(id))")
                }
            } else {
                logger.warning("No paired devices found")
            }
        } catch {
            logger.error("Failed to get paired devices: \(error)")
        }

        var attemptingConnection = false

        try await NetworkBrowser(for:
            .wifiAware(.connecting(to: .allPairedDevices, from: .simulationService))
        )
        .onStateUpdate { browser, state in
            logger.info("\(String(describing: browser)): \(String(describing: state))")

            switch state {
            case .setup, .waiting: break
            case .ready: self.localEventsContinuation.yield(.browserRunning)
            case .failed, .cancelled: self.localEventsContinuation.yield(.browserStopped)
            default: break
            }
        }
        .run { waEndpoints in
            logger.info("Discovered: \(waEndpoints)")
            if let endpoint = waEndpoints.first, !attemptingConnection {
                attemptingConnection = true
                await self.setupConnection(to: endpoint)
            }
        }
    }

// MARK: - Send

    func send(_ event: NetworkEvent, to connection: WiFiAwareConnection) async {
        await connectionManager.send(event, to: connection)
    }

    func sendToAll(_ event: NetworkEvent) async {
        await connectionManager.sendToAll(event)
    }

// MARK: - Chunked File Transfer
    
    func sendFileToAll(_ data: Data, filename: String) async {
        await connectionManager.sendFileToAll(data, filename: filename)
    }

// MARK: - Deinit

    deinit {
        localEventsContinuation.finish()
        networkEventsContinuation.finish()
    }
}

public enum NetworkEvent: Codable, Sendable {
    case startStreaming
    case satelliteMovedTo(position: CGPoint, dimensions: CGSize)
    
    // New cases for chunked file transfer
    case startFileTransfer(filename: String, chunkSize: Int, totalChunks: Int, battery: [String: String], receiver: String)
    case fileChunk(data: Data, chunkIndex: Int, filename: String, timestamp: Date)
    case endFileTransfer(filename: String, latency: Double)
    
    // RTT measurement events
    case ping(sequenceNumber: Int)
    case pong(sequenceNumber: Int)
    
    // Configuration change events
    case configurationChange(performanceMode: WAPerformanceMode, accessCategory: WAAccessCategory)
}
