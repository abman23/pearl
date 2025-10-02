/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The main top-level view to select the app mode.
*/

import SwiftUI
import OSLog

struct ContentView: View {
    @State var showSimulationScene: Bool = false
    @State var selectedMode: SimulationEngine.Mode?

    var body: some View {
        VStack {
            if showSimulationScene, let selectedMode {
                SimulationView(mode: selectedMode)
            } else {
                Spacer()

                MenuButton(mode: .host, selectedMode: $selectedMode, showSimulationScene: $showSimulationScene)
                MenuButton(mode: .viewer, selectedMode: $selectedMode, showSimulationScene: $showSimulationScene)

                Spacer()
            }
        }
    }
}

struct MenuButton: View {
    let mode: SimulationEngine.Mode

    @Binding var selectedMode: SimulationEngine.Mode?
    @Binding var showSimulationScene: Bool

    var body: some View {
        Button {
            logger.info("Selected Mode: \(mode)")
            selectedMode = mode
            showSimulationScene = true
        } label: {
            Label(label, systemImage: image)
                .foregroundStyle(.white)
                .font(.largeTitle)
        }
        .tint(.blue)
        .buttonStyle(.borderedProminent)
        .controlSize(.extraLarge)
        .padding([.bottom], 15.0)
    }

    var label: String {
        switch mode {
            case .host: return "Host Simulation"
            case .viewer: return "View Simulation"
        }
    }

    var image: String {
        switch mode {
            case .host: return "server.rack"
            case .viewer: return "eye"
        }
    }
}

#Preview {
    ContentView()
}
