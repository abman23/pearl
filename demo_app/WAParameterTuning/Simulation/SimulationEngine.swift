/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Manages the simulation state.
*/

import Foundation
import Observation
import SpriteKit
import WiFiAware
import Network
import SwiftUI
import OSLog
import FoundationModels

@MainActor @Observable class SimulationEngine {
    private let mode: Mode
    private var scene: SimulationScene?

    var networkState: NetworkState = .notStarted
    var deviceConnections: [WAPairedDevice: ConnectionDetail] = [:]

    private let connectionManager: ConnectionManager
    private let networkManager: NetworkManager

    @ObservationIgnored var networkTask: Task<Void, Error>?
    @ObservationIgnored private var simulationEventsTask: Task<Void, Error>?
    @ObservationIgnored private var monitorTimer: Timer?
    @ObservationIgnored private var fileTransmissionTimer: Timer?
    
    // File transfer state management
    private var logFile: String = addPrefix(to: "contexts", ext: "json")
    private var incomingFiles: [String: IncomingFile] = [:]
    var contextsList: [[String: Any]] = []
    var transmissionCount = 0
    let maxTransmissions = 2000
    var contextsLog4AFM: [[String: String]] = []
    var AFMparameter: (String, String) = (appPerformanceMode.descirption, appAccessCategory.description)
    var AFMResponse: String = ""
    var contextsLogStrList: [String] = []
    var logStore = LogStore.shared
    
    private struct IncomingFile {
        var chunks: [Int: Data] = [:]
        var chunkSize: Int
        var totalChunks: Int
        var battery: [String: String]
        var receivedChunks: Set<Int> = []
        var chunkLatencies: [Double] = []
        var receiver: String
    }

    init(mode: Mode) async {
        self.mode = mode

        connectionManager = await ConnectionManager()
        networkManager = NetworkManager(connectionManager: connectionManager)

        await withTaskGroup { group in
            group.addTask {
                await self.setupEventProcessing(for: self.networkManager.localEvents)
            }
            group.addTask {
                await self.setupEventProcessing(for: self.networkManager.networkEvents)
            }
            group.addTask {
                await self.setupEventProcessing(for: self.connectionManager.localEvents)
            }
            group.addTask {
                await self.setupEventProcessing(for: self.connectionManager.networkEvents)
            }

            group.cancelAll()
        }

        startConnectionMonitor(interval: 3.0)
//        if mode == .viewer {
//            startFileTransmission(interval: 10.0)
//        }
    }

    func setupEventProcessing<T>(for stream: AsyncStream<T>) -> Task<Void, Error> {
        return Task {
            for await event in stream {
                if T.self == LocalEvent.self {
                    await processLocalEvent(event as? LocalEvent)
                } else if T.self == NetworkEvent.self {
                    await processNetworkEvent(event as? NetworkEvent)
                }
            }
        }
    }

    func startConnectionMonitor(interval: TimeInterval) {
        monitorTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            Task { [weak self] in
                try await self?.connectionManager.monitor()
            }
        }
    }
    
    func startFileTransmission(interval: TimeInterval) {
        transmissionCount = 0
        logFile = addPrefix(to: "contexts", ext: "json")
        logger.info("Start file transmission with an interval of \(interval)s")
        
        fileTransmissionTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            Task { [weak self] in
                let filename: String = UUID().uuidString
                let data = createDummyFile(sizeInByte: 1000)
                await self?.sendFile(data: data, filename: filename)
            }
            
            Task { @MainActor in
                self.transmissionCount += 1
                logger.info("\(self.transmissionCount) / \(self.maxTransmissions) files transmitted")
                if self.transmissionCount >= self.maxTransmissions {
                    logger.info("File transmission completed after \(self.maxTransmissions) runs")
                    self.fileTransmissionTimer?.invalidate()
                    self.fileTransmissionTimer = nil
                }
            }
        }
    }

    func setup(with scene: SimulationScene) {
        self.scene = scene
        scene.setup(as: mode)

        guard let simulationEvents = scene.localEvents else { return }
        simulationEventsTask = Task {
            for await event in simulationEvents {
                await processLocalEvent(event)
            }
        }
    }

    func processLocalEvent(_ event: LocalEvent?) async {
        guard let event else { return }

        switch event {
        case .browserRunning, .listenerRunning:
            networkState = .running

        case .browserStopped, .listenerStopped:
            networkState = .stopped

        case .connectionReady(let device, let connectionInfo):
            deviceConnections[device] = connectionInfo
            logStore.add("Successfully built WA connection with subscriber \(device.displayName). Current WA parameter tuple: \(AFMparameter).")
            if mode == .viewer {
                networkTask?.cancel()
                networkTask = nil
                scene?.enableSatellite()

                await networkManager.send(.startStreaming, to: connectionInfo.connection)
            }

        case .connectionStopped(let device, let connectionID):
            deviceConnections.removeValue(forKey: device)
            await connectionManager.invalidate(connectionID)
            if mode == .viewer {
                networkState = .stopped
                scene?.disableSatellite()
            }

        case .connectionPerformanceUpdate(let device, let connectionInfo):
            deviceConnections[device] = connectionInfo

        case .satelliteMovedTo(let position):
            if mode == .host, let scene {
                await networkManager.sendToAll(.satelliteMovedTo(position: position, dimensions: scene.frame.size))
            }
        }
    }
    
    func changeConfiguration(performanceMode: WAPerformanceMode, accessCategory: WAAccessCategory) async {
//        logger.info("SimulationEngine: Changing configuration to \(performanceMode.descirption), \(accessCategory.description)")
        
        // Update NetworkManager configuration
        let success: Bool = await networkManager.changeConfiguration(performanceMode: performanceMode, accessCategory: accessCategory, mode: mode)
        print("Success: \(success)")
        
        // Notify all connected devices about the configuration change
        if success {
            let configEvent = NetworkEvent.configurationChange(performanceMode: performanceMode, accessCategory: accessCategory)
            self.AFMparameter = (performanceMode.descirption, accessCategory.description)
            await connectionManager.sendToAll(configEvent)
//            logStore.add("Notified subscriber to update WA parameters.")
            
            if mode == .host {
                //            try? await Task.sleep(nanoseconds: 3_000_000_000) // wait 1s for the previous connection to be cancelled
//                logger.info("Restarting listen with configuration")
                networkTask?.cancel()
                run()
                logStore.add("Restarted listener with new WA parameter tuple (\(performanceMode.descirption), \(accessCategory.description)). Waiting for subscriber to updated and reconnected.")
            }
        } else {
            logStore.add("New WA parameters suggested by AFM are the same as current ones. No adjustment needed.")
//            logger.info("New parameters are the same as current ones. No adjustment needed.")
        }
    }

    func processNetworkEvent(_ event: NetworkEvent?) async {
        guard let event else { return }

        switch event {
        case .startStreaming: logger.info("Received Start streaming")
        case .satelliteMovedTo(position: let position, dimensions: let dimensions):
            if mode == .viewer {
                scene?.moveSatellite(to: position, using: dimensions)
            }
            
        // Handle chunked file transfer events
        case .startFileTransfer(let filename, let chunkSize, let totalChunks, let battery, let receiver):
            incomingFiles[filename] = IncomingFile(chunkSize: chunkSize, totalChunks: totalChunks, battery: battery, receiver: receiver)
//            logger.info("Start receiving chunked file: \(filename), chunk size: \(chunkSize), total chunks: \(totalChunks)")
            
        case .fileChunk(let data, let chunkIndex, let filename, let sendTime):
            let receiveTime: Date = Date()
            guard var incomingFile = incomingFiles[filename] else {
//                logger.error("Received chunk for unknown file: \(filename)")
                return
            }
            
            incomingFile.chunks[chunkIndex] = data
            incomingFile.receivedChunks.insert(chunkIndex)
            
            // calculate latency
            let latency = receiveTime.timeIntervalSince(sendTime) * 1000 // Convert to milliseconds
            incomingFile.chunkLatencies.append(latency)
            incomingFiles[filename] = incomingFile
            
//            logger.info("Received chunk \(chunkIndex) for \(filename), latency: \(String(format: "%.2f", latency))ms, progress: \(incomingFile.receivedChunks.count)/\(incomingFile.totalChunks)")
            
            
        case .endFileTransfer(let filename, let senderLatency):
            guard let incomingFile: IncomingFile = incomingFiles[filename] else {
//                logger.error("End transfer for unknown file: \(filename)")
                return
            }
            
            // Check if all chunks received
            if incomingFile.receivedChunks.count == incomingFile.totalChunks {
                // Reassemble file
                var reassembledData = Data()
                for chunkIndex in 0..<incomingFile.totalChunks {
                    if let chunkData = incomingFile.chunks[chunkIndex] {
                        reassembledData.append(chunkData)
                    }
                }
                
//                logger.info("File transfer completed: \(filename), reassembled size: \(reassembledData.count) bytes")
                logStore.add("Interacting with subscriber. \(contextsList.count+1) events delivered.")
                await handleReceivedFile(data: reassembledData, filename: filename, incomingFile: incomingFile, senderLatency: senderLatency, logFile: logFile)
                
                // Clean up
                incomingFiles.removeValue(forKey: filename)
            } else {
//                logger.error("File transfer incomplete: \(filename), received \(incomingFile.receivedChunks.count)/\(incomingFile.totalChunks) chunks")
                incomingFiles.removeValue(forKey: filename)
            }
            
        // Handle configuration change events
        case .configurationChange(let performanceMode, let accessCategory):
            logger.info("Received configuration change: \(performanceMode.descirption), \(accessCategory.description)")
            
            // Update local configuration
            await networkManager.changeConfiguration(performanceMode: performanceMode, accessCategory: accessCategory, mode: mode)
            self.AFMparameter = (performanceMode.descirption, accessCategory.description)
            
            if mode == .viewer {
                for device in deviceConnections.keys {
                    logger.info("Stopping old connection to \(device.displayName)")
                    await stopConnection(to: device)
                }
                
                // Wait for listener to be set up
                try? await Task.sleep(nanoseconds: 5_000_000_000) // 5s
                
                logger.info("Restarting browser to connect with new configuration")
                networkTask?.cancel()
                run()
            }
            
        case .ping, .pong:
            // These are handled in ConnectionManager
            break
        }
    }
    
    private func handleReceivedFile(data: Data, filename: String, incomingFile: IncomingFile, senderLatency: Double, logFile: String) async {
//        logger.info("Received file: \(filename), size: \(data.count) bytes")
        
        // Store file in documents directory, uncomment for data collection
//        do {
//            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
//            let fileURL = documentsPath.appendingPathComponent(filename)
//            try data.write(to: fileURL)
//            logger.info("File saved to: \(fileURL.path)")
//        } catch {
//            logger.error("Failed to save video: \(error)")
//        }
        
        // Store context information as a JSON file in documents directory
        var contexts: [String: Any] = [:]
        var contextsEntry: [String: String] = [:]
        let dateFormatter: DateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        
        guard let connectionDetail: ConnectionDetail = deviceConnections.first?.value else {
//            logger.error("No connection details found")
            return
        }
        
        let performanceReport: WAPerformanceReport = connectionDetail.performanceReport
        let devicePerformance: [String: Any] = ["report_timestamp": dateFormatter.string(from: performanceReport.timestamp),
                                                "throughputCapacityRatio": performanceReport.throughputCapacityRatio ?? -1.0, "throughputCeiling": performanceReport.throughputCeiling ?? -1.0,
                                                "throughputCapacity": performanceReport.throughputCapacity ?? -1.0, "signalStrength": performanceReport.signalStrength ?? -1.0,
                                                "latency": performanceReport.transmitLatency[appAccessCategory]?.average?.milliseconds ?? -1.0]
        
        // Latency measured by RTT-based mrthod
//        contexts["senderLatency"] = senderLatency
//        if let connection: WiFiAwareConnection = deviceConnections.first?.value.connection {
//            let rtt: Double = await connectionManager.measureRTT(to: connection) ?? -2.0
//            contexts["rttLatency"] = rtt / 2.0
//        }
        
        contexts["senderDevice"] = getDeviceInfo(device: deviceConnections.first?.key)
        contexts["receiverDevice"] = incomingFile.receiver
            
        contexts["performanceReport"] = devicePerformance
        
        let timestamp: String = dateFormatter.string(from: Date())
        contexts["timestamp"] = timestamp
        contextsEntry["time"] = String(timestamp.split(separator: " ")[1])
        
        
        let dataTypes: [String] = sampleDataTypes(for: Date(), number: 1)
        contexts["applicationTypes"] = dataTypes
        contextsEntry["dataType"] = dataTypes[0]
        
        let localBattery: [String: String] = getBatteryInfo()
        contexts["localBattery"] = localBattery
        contextsEntry["localBattery"] = localBattery["batteryLevel"]
        
        let remoteBattery: [String: String] = incomingFile.battery
        contexts["remoteBattery"] = remoteBattery
        contextsEntry["remoteBattery"] = remoteBattery["batteryLevel"]
        
        contexts["performanceMode"] = await networkManager.currentPerformanceMode.descirption
        contexts["accessCategory"] = await networkManager.currentAccessCategory.description
        
        // Save the contexts
        print(contexts)
        appendDataPointToJSONFile(dataPoint: contexts, filename: logFile)
        contextsList.append(contexts)
        contextsLog4AFM.append(contextsEntry)
        
        let localBatteryLevel: String = contextsEntry["localBattery"] ?? "0.0"
        let remoteBatteryLevel: String = contextsEntry["remoteBattery"] ?? "0.0"
        let time: String = contextsEntry["time"] ?? "unknown time"
        let application: String = contextsEntry["dataType"] ?? "unknown app"
        
        let contextsEntryStr = "| \(time) | \(application) | \(localBatteryLevel) | \(remoteBatteryLevel) |"
        contextsLogStrList.append(contextsEntryStr)
         
        // Call on-device LLM to adjust WA parameter
        if contextsList.count % 10 == 0 && contextsList.count > 0 {
            let contextsCopy = await MainActor.run { self.contextsLog4AFM }
            let (parameterTuple, responseText) = await ModelInterface.modelCompletion(contextsList: contextsCopy)
            // No adjustment if no valid response
            if parameterTuple != ("unknownMode", "unknownCategory") {
                logStore.add("Received WA parameter tuple suggested by AFM: \(parameterTuple).")
                let (mode, category) = matchWAParameterTuple(stringParamTuple: parameterTuple)
                await changeConfiguration(performanceMode: mode, accessCategory: category)
            } else {
                logStore.add("No valid WA parameter tuple suggested by AFM.")
            }
            self.AFMResponse = responseText
        }
        
    }

    func sendFile(data: Data, filename: String) async {
        await networkManager.sendFileToAll(data, filename: filename)
    }

    func run() {
//    func run() -> Task<Void, Error>? {
        networkTask = Task {
            _ = try await withTaskCancellationHandler {
                switch mode {
                case .host: try await networkManager.listen()
                case .viewer: try await networkManager.browse()
                }
            } onCancel: {
                Task { @MainActor in
                    networkState = .stopped
                }
            }
        }

//        return networkTask
    }

    func stopConnection(to device: WAPairedDevice) async {
        if let connection = deviceConnections[device]?.connection {
            await connectionManager.stop(connection)
        } else {
            logger.error("Unable to find the connection for \(device)")
        }
    }

    nonisolated func stopConnectionMonitor() {
        Task { @MainActor in
            monitorTimer?.invalidate()
            monitorTimer = nil
        }
    }
    
    nonisolated func stopFileTransmission() {
        Task { @MainActor in
            fileTransmissionTimer?.invalidate()
            fileTransmissionTimer = nil
        }
    }

    deinit {
        simulationEventsTask?.cancel()
        simulationEventsTask = nil

        networkTask?.cancel()
        networkTask = nil

        stopConnectionMonitor()
        stopFileTransmission()
    }
}

struct ConnectionDetail: Sendable, Equatable {
    let connection: WiFiAwareConnection
    let performanceReport: WAPerformanceReport

    public static func == (lhs: ConnectionDetail, rhs: ConnectionDetail) -> Bool {
        return lhs.performanceReport.localTimestamp == rhs.performanceReport.localTimestamp
    }
}

enum LocalEvent: Sendable {
    case browserRunning
    case browserStopped

    case listenerRunning
    case listenerStopped

    case connectionReady(WAPairedDevice, ConnectionDetail)
    case connectionStopped(WAPairedDevice, WiFiAwareConnectionID)
    case connectionPerformanceUpdate(WAPairedDevice, ConnectionDetail)
    case satelliteMovedTo(CGPoint)
}
