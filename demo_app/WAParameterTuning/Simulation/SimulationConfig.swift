//
//  SimulationConfig.swift
//  WAParameterTuning
//
//  Created by Yanqing Lu on 7/1/25.
//  Copyright © 2025 Apple. All rights reserved.
//

import Foundation

@MainActor let dataTypeProfiles: [String: [String: Any]] = [
    "textMessage": [
        "latencyTolerance": 100,
        "throughputRequirement": 10,
        "priorityWeight": 1.0,
        "preferredMode": "realtime",
        "preferredCategory": "interactiveVoice"
    ],
    "voiceChat": [
        "latencyTolerance": 50,
        "throughputRequirement": 64,
        "priorityWeight": 1.5,
        "preferredMode": "realtime",
        "preferredCategory": "interactiveVoice"
    ],
    "videoCall": [
        "latencyTolerance": 100,
        "throughputRequirement": 1500,
        "priorityWeight": 2.0,
        "preferredMode": "realtime",
        "preferredCategory": "interactiveVideo"
    ],
    "sensorSync": [
        "latencyTolerance": 1000,
        "throughputRequirement": 50,
        "priorityWeight": 0.5,
        "preferredMode": "bulk",
        "preferredCategory": "background"
    ],
    "photoTransfer": [
        "latencyTolerance": 2000,
        "throughputRequirement": 10000,
        "priorityWeight": 1.0,
        "preferredMode": "bulk",
        "preferredCategory": "bestEffort"
    ],
    "videoUpload": [
        "latencyTolerance": 5000,
        "throughputRequirement": 30000,
        "priorityWeight": 1.5,
        "preferredMode": "bulk",
        "preferredCategory": "bestEffort"
    ],
    "firmwareUpdate": [
        "latencyTolerance": 10000,
        "throughputRequirement": 20000,
        "priorityWeight": 1.0,
        "preferredMode": "bulk",
        "preferredCategory": "background"
    ],
    "mapSync": [
        "latencyTolerance": 500,
        "throughputRequirement": 500,
        "priorityWeight": 1.2,
        "preferredMode": "realtime",
        "preferredCategory": "bestEffort"
    ]
]

let timeBasedUsageProbabilities: [Range<Int>: [String: Double]] = [
    0..<6: [
        "firmwareUpdate": 0.4,
        "sensorSync": 0.3,
        "textMessage": 0.1,
        "photoTransfer": 0.1,
        "mapSync": 0.1
    ],
    6..<12: [
        "textMessage": 0.25,
        "voiceChat": 0.2,
        "videoCall": 0.25,
        "mapSync": 0.2,
        "photoTransfer": 0.1
    ],
    12..<18: [
        "textMessage": 0.2,
        "voiceChat": 0.2,
        "videoCall": 0.25,
        "sensorSync": 0.2,
        "photoTransfer": 0.15
    ],
    18..<24: [
        "textMessage": 0.1,
        "voiceChat": 0.2,
        "videoUpload": 0.3,
        "photoTransfer": 0.2,
        "videoCall": 0.2
    ]
]


func sampleDataTypes(for date: Date = Date(), number: Int) -> [String] {
    var dataTypes: [String] = []
    let hour = Calendar.current.component(.hour, from: date)
    
    // Find the matching time block
    guard let timeBlock = timeBasedUsageProbabilities.first(where: { $0.key.contains(hour) })?.value else {
        return dataTypes
    }

    // Create cumulative distribution
    let total = timeBlock.values.reduce(0, +)
    let normalized = timeBlock.mapValues { $0 / total }
    
    for _ in 0..<number {
        let random = Double.random(in: 0..<1)
//        print("Random: \(random)")
        var cumulative = 0.0
        for (type, prob) in normalized {
            cumulative += prob
            if random <= cumulative {
                dataTypes.append(type)
                break
            }
        }
    }
    
    return dataTypes
}

