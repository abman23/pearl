/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The entry point for the app.
*/

import SwiftUI
import WiFiAware
import OSLog
import FoundationModels

let logger = Logger(subsystem: "yqlu.WAParameterTuning", category: "App")
let model = SystemLanguageModel.default



@main
struct WAParameterTuningApp: App {
    var body: some Scene {
        WindowGroup {
            if WACapabilities.supportedFeatures.contains(.wifiAware) && model.isAvailable {
                ContentView()
            } else if (model.isAvailable) {
                ContentUnavailableView {
                    Label("This device does not support Wi-Fi Aware", systemImage: "exclamationmark.octagon")
                }
            } else {
                switch model.availability {
                    case .available:
                        Text("On-device LLM is available")
                    case .unavailable(.deviceNotEligible):
                        // Show an alternative UI.
                        Text("On-device LLM is unavailable - device not eligible")
                    case .unavailable(.appleIntelligenceNotEnabled):
                        // Ask the person to turn on Apple Intelligence.
                        Text("On-device LLM is unavailable - apple intelligence not enabled")
                    case .unavailable(.modelNotReady):
                        // The model isn't ready because it's downloading or because of other system reasons.
                        Text("On-device LLM is unavailable - model not ready")
                    case .unavailable(_):
                        // The model is unavailable for an unknown reason.
                        Text("On-device LLM is unavailable - other reasons")
                }
            }
        }
    }
}
