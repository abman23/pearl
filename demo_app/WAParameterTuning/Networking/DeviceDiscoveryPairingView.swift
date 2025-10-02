/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The DeviceDiscoveryUI views for pairing.
*/

import DeviceDiscoveryUI
import WiFiAware
import SwiftUI
import Network
import OSLog

struct DeviceDiscoveryPairingView: View {
    let mode: SimulationEngine.Mode

    var body: some View {
        if mode == .viewer {
            DevicePicker(.wifiAware(.connecting(to: .selected([]), from: .simulationService))) { endpoint in
                logger.info("Paired Endpoint: \(endpoint)")
            } label: {
                Image(systemName: "plus")
//                Text("Add Device")
            } fallback: {
                Image(systemName: "xmark.circle")
//                Text("Unavailable")
            }
            .buttonStyle(.borderedProminent).controlSize(.extraLarge)
            .padding()
        } else {
            DevicePairingView(.wifiAware(.connecting(to: .simulationService, from: .selected([])))) {
                Image(systemName: "plus")
//                Text("Add Device")
            } fallback: {
                Image(systemName: "xmark.circle")
//                Text("Unavailable")
            }
            .buttonStyle(.borderedProminent).controlSize(.extraLarge)
            .padding()
        }
    }
}
