/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Extensions to the WiFiAware framework.
*/

import WiFiAware
import Network

//let simulationServiceName = "_file-service._udp"
let simulationServiceName = "_file-service._tcp"

extension WAPublishableService {
    public static var simulationService: WAPublishableService {
        allServices[simulationServiceName]!
    }
}

extension WASubscribableService {
    public static var simulationService: WASubscribableService {
        allServices[simulationServiceName]!
    }
}

extension WAPerformanceMode {
    var descirption: String {
        switch self {
        case .bulk: "bulk"
        case .realtime: "realtime"
        default : "unknown"
        }
    }
}

extension WAAccessCategory {
    var serviceClass: NWParameters.ServiceClass {
        switch self {
        case .bestEffort: .bestEffort
        case .background: .background
        case .interactiveVideo: .interactiveVideo
        case .interactiveVoice: .interactiveVoice
        default : .bestEffort
        }
    }
    
    var description: String {
        switch self {
        case .bestEffort: "bestEffort"
        case .background: "background"
        case .interactiveVideo: "interactiveVideo"
        case .interactiveVoice: "interactiveVoice"
        default : "unknown"
        }
    }
}

extension WAPairedDevice {
    var displayName: String {
        let displayName = self.name ?? self.pairingInfo?.pairingName ?? ""
        return "\(displayName) (\(self.pairingInfo?.modelName ?? "unknown model"), \(self.pairingInfo?.vendorName ?? "unknown vendor"))"
    }
    
    var modelName: String {
        return self.pairingInfo?.modelName ?? "unknown model"
    }
    
    var vendorName: String {
        return self.pairingInfo?.vendorName ?? "unknown vendor"
    }
}

extension WAPerformanceReport {
    var display: String {
        return "[\(self.timestamp)] Signal Strength: \(self.signalStrength?.description ?? "-"), Throughput Capacity Ratio: \(self.throughputCapacityRatio?.description ?? "-"), Transmit Latency: \(self.transmitLatency)"
    }
}
