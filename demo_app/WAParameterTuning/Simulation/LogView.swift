//
//  LogView.swift
//  WAParameterTuning
//
//  Created by Yanqing Lu on 8/20/25.
//  Copyright © 2025 Apple. All rights reserved.
//

import SwiftUI
import OSLog

struct LogView: View {
    var logStore = LogStore.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("System log (Wi-Fi Aware):")
                .fontWeight(.bold)
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading) {
                        ForEach(logStore.messages.indices, id: \.self) { idx in
                            HStack(alignment: .top, spacing: 8) {
                                Text("\(idx + 1)")
                                    .font(.system(.footnote, design: .monospaced))
                                    .foregroundColor(.secondary)
                                    .frame(width: 20, alignment: .trailing)
                                
                                Text(logStore.messages[idx])
                                    .font(.system(.footnote, design: .monospaced))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
//                            .padding(.vertical, 2)
                        }
                    }
                    .padding()
                }
                .background(Color.black.opacity(0.05))
                .onChange(of: logStore.messages.count) { _, _ in
                    if let lastIndex = logStore.messages.indices.last {
                        withAnimation(.easeOut(duration: 0.3)) {
                            proxy.scrollTo(lastIndex, anchor: .bottom)
                        }
                    }
                }
            }
        }
    }
}


@Observable class LogStore {
    @MainActor static let shared = LogStore()
    
    var messages: [String] = []
    
    func add(_ message: String) {
        self.messages.append(message)
    }
}
