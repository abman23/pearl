# WA Parameter Tuning using On-device LLM

This demo app builds Wi-Fi Aware (WA) connection between two devices and simulate their interactions with synthetic applications. It also uses Apple's on-device LLM to adjust WA parameter. The code is adapted from Apple's [Wi-Fi Aware sample app](https://developer.apple.com/documentation/wifiaware/building-peer-to-peer-apps).

## Requirements
- This app can only be run on **physical** [iPhone or iPad](https://developer.apple.com/documentation/wifiaware) devices with iOS 26.0+ or iPadOS 26.0+. Simulator is not supported. Building this project also requires Xcode 26.0+.
- Wi-Fi Aware framework used in this app requires an Apple Developer account which has joined in [Apple Developer Program](https://developer.apple.com/programs/).
- The device must support Apple Intelligence and have it turned on in Settings (it can take several minutes for the model to download and become available when a person turns on Apple Intelligence).

## Code Structure Overview
```
├── WAParameterTuning
│   ├── Assets.xcassets
│   ├── ContentView.swift  # The main top-level view to select the app mode
│   ├── Extensions
│   │   ├── SimulationEngine+Extensions.swift
│   │   └── WiFiAware+Extensions.swift
│   ├── LanguageModel
│   │   └── ModelInterface.swift  # Interaction with on-device LLM
│   ├── Networking
│   │   ├── ConnectionManager.swift
│   │   ├── DeviceDiscoveryPairingView.swift  
│   │   ├── NetworkConfig.swift
│   │   └── NetworkManager.swift  # Control WA connections
│   ├── PairedDevices
│   │   └── PairedDevicesView.swift
│   ├── README.md
│   ├── Resources
│   │   ├── Info.plist
│   │   └── WAParameterTuningApp.entitlements
│   ├── Simulation
│   │   ├── LogView.swift
│   │   ├── SimulationConfig.swift
│   │   ├── SimulationEngine.swift  # Manage simulation and on-device LLM
│   │   ├── SimulationScene.swift
│   │   ├── SimulationView.swift
│   │   └── Utils.swift
│   └── WAParameterTuningApp.swift  # Entry point of the app
└── WAParameterTuning.xcodeproj
```
