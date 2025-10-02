/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The simulation view containing the simulation scene, the buttons to start and stop networking, and the paired devices list.
*/

import SwiftUI
import SpriteKit
import OSLog
import WiFiAware

struct SimulationView: View {
    private let mode: SimulationEngine.Mode
    

    var scene = SimulationScene()
    @State private var engine: SimulationEngine?
    @State private var networkTask: Task<Void, Error>?
    @State private var isSending = false
    @State private var sendStatus: String? = nil  // "success" or "fail"
    
    @State private var selectedPerformanceMode: WAPerformanceMode = appPerformanceMode
    @State private var selectedAccessCategory: WAAccessCategory = appAccessCategory
    
    @State private var performanceModeChanged = false
    @State private var accessCategoryChanged = false
    @State private var afmResponseChanged = false
    
    init(mode: SimulationEngine.Mode) {
        self.mode = mode
    }

    var body: some View {
        GeometryReader { geoReader in
            VStack {
                if let engine {
                    HStack {
                        DeviceDiscoveryPairingView(mode: mode)

                        Spacer()

                        HStack(spacing: 0) {
                            let afm = engine.AFMparameter
                            Text("PerformanceMode=")
                                .font(.system(size: 17, design: .default).bold())
                            Text("{\(afm.0)}")
                                .font(.system(size: 17, design: .monospaced).bold())
                                .foregroundColor(.red)
                                .background(
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(Color.red.opacity(0.4))
                                        .opacity(performanceModeChanged ? 1 : 0)
                                        .animation(.easeInOut(duration: 1.0), value: performanceModeChanged)
                                )
                                .onChange(of: afm.0) { _, _ in
                                    performanceModeChanged = true
                                    // Reset highlight after animation
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                                        performanceModeChanged = false
                                    }
                                }

                            Text(", AccessCategory=")
                                .font(.system(size: 17, design: .default).bold())
                            Text("{\(afm.1)}")
                                .font(.system(size: 17, design: .monospaced).bold())
                                .foregroundColor(.red)
                                .background(
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(Color.red.opacity(0.4))
                                        .opacity(accessCategoryChanged ? 1 : 0)
                                        .animation(.easeInOut(duration: 1.0), value: accessCategoryChanged)
                                )
                                .onChange(of: afm.1) { _, _ in
                                    accessCategoryChanged = true
                                    // Reset highlight after animation
                                    if accessCategoryChanged {
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                                            accessCategoryChanged = false
                                        }
                                    }
                                }
                        }
                        
                        
                        Spacer()

                        Button {
                            if engine.networkState == .running {
                                logger.info("Cancel Network Task")
                                engine.networkTask?.cancel()
                                engine.networkTask = nil
                            } else {
                                logger.info("Start Network Task")
                                engine.run()
                            }
                        } label: {
                            let buttonImage = Image(systemName: "dot.radiowaves.left.and.right")
                            if engine.networkState == .running {
                                buttonImage.symbolEffect(.variableColor.cumulative.dimInactiveLayers.reversing, options: .repeat(.continuous))
                            } else {
                                buttonImage.symbolEffectsRemoved()
                            }

//                            Text(engine.networkState.description(for: mode, isConnected: !engine.deviceConnections.isEmpty))
                        }
                        .buttonStyle(.borderedProminent).controlSize(.extraLarge)
                        .tint(engine.networkState == .running ? .green : .blue)
                        .padding()
                        .disabled(mode == .viewer && !engine.deviceConnections.isEmpty)
                    }
                    
                    // Start interaction simulation
                    if mode == .viewer {
                        Button {
                            Task {
                                isSending = true
                                engine.startFileTransmission(interval: 3)
                                isSending = false
                            }
                        } label: {
                            HStack {
                                Image(systemName: "video")
                                Text(buttonText(for: isSending, engine: engine))
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .tint(buttonColor(isSending: isSending))
                        .disabled(isSending)
                        .padding()
                    }
                    
                    if mode == .host {
                        // Show contexts and responses
                        HStack {
                            VStack(alignment: .leading) {
                                // Contexts log section
                                Text("Input (context info):")
                                    .fontWeight(.bold)
                                Text("  | Time of day | Application | Publisher Battery Level | Subscriber Battery Level |")
                                    .font(.system(.footnote, design: .monospaced))
                                
                                ScrollViewReader { proxy in
                                    ScrollView {
                                        VStack(alignment: .leading) {
                                            ForEach(engine.contextsLogStrList.indices, id: \.self) { idx in
                                                Text(engine.contextsLogStrList[idx])
                                                    .font(.system(.footnote, design: .monospaced))
                                                    .frame(maxWidth: .infinity, alignment: .leading)
                                                    .id(idx)
                                            }
                                        }
                                        .padding()
                                    }
                                    .background(Color.black.opacity(0.05))
                                    .onChange(of: engine.contextsLogStrList.count) { _, _ in
                                        if let lastIndex = engine.contextsLogStrList.indices.last {
                                            withAnimation(.easeOut(duration: 0.3)) {
                                                proxy.scrollTo(lastIndex, anchor: .bottom)
                                            }
                                        }
                                    }
                                }
                                
                                // AFM response section
                                Text("Output (on-device LLM response):")
                                    .fontWeight(.bold)
                                Text(engine.AFMResponse)
                                .font(.system(.footnote, design: .monospaced))
//                                .background(Color.black.opacity(0.05))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(Color.blue.opacity(0.3))
                                        .opacity(afmResponseChanged ? 1 : 0)
                                        .animation(.easeInOut(duration: 1.5), value: afmResponseChanged)
                                )
                                .onChange(of: engine.AFMResponse) { _, _ in
                                    afmResponseChanged = true
                                    // Reset highlight after animation
                                    if afmResponseChanged {
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                                            afmResponseChanged = false
                                        }
                                    }
                                }
                                
                            }
//                            .frame(maxWidth: .infinity, alignment: .leading)
                            
                            VStack(alignment: .leading) {
                                LogView()
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }

                    SpriteView(scene: scene, debugOptions: [.showsFPS])

                    PairedDevicesView(engine: engine)
                        .frame(height: 0.05 * geoReader.size.height)
                }
            }
            .task {
                let minDimension = min(geoReader.size.width, geoReader.size.height)
                scene.size = .init(width: minDimension, height: minDimension)
                scene.scaleMode = .aspectFill

                engine = await .init(mode: mode)
                engine?.setup(with: scene)
            }
        }
    }
    
    func sendSampleFile(engine: SimulationEngine?) async -> Bool {
        guard let engine = engine else {
            logger.error("Engine is nil.")
            return false
        }
        let name: String = "Assignment1_video", ext: String = ".mov"
        var filename: String = name + ext
        

        filename = "dummy_file_1b.bin"
        let data = createDummyFile(sizeInByte: 1000)
        await engine.sendFile(data: data, filename: filename)
//                logger.info("Sent sample file: \(filename)")
        return true
    }
    
    func buttonText(for isSending: Bool, engine: SimulationEngine) -> String {
        if isSending {
            return "Sending..."
        } else {
//            return "Send File: \(engine.transmissionCount)/\(engine.maxTransmissions)"
            return "Start Interaction"
        }
    }

    func iconName(for status: String?) -> String {
        switch status {
        case "success": return "checkmark"
        case "fail": return "xmark.octagon"
        default: return "video"
        }
    }
    
    func buttonColor(isSending: Bool) -> Color {
        if isSending {
            return .gray
        } else {
            return .purple
        }
    }
}
