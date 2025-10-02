//
//  Utils.swift
//  WAFileTransmission
//
//  Created by Yanqing Lu on 6/24/25.
//
import SwiftUI
import OSLog
import WiFiAware
import Foundation

func appendDataPointToJSONFile(dataPoint: [String: Any], filename: String) {
    let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let fileURL = documentsPath.appendingPathComponent(filename)
    
    // Check if the file exists
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: fileURL.path) {
        // If the file doesn't exist, create an empty file
        fileManager.createFile(atPath: fileURL.path, contents: nil, attributes: nil)
    }
    
    // Serialize the data point to JSON
    do {
        let jsonData = try JSONSerialization.data(withJSONObject: dataPoint, options: [])
        
        // Convert the JSON data to a string
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            // Append the string to the file
            if let fileHandle = FileHandle(forWritingAtPath: fileURL.path) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(jsonString.data(using: .utf8)!)
                fileHandle.write("\n".data(using: .utf8)!) // Add a newline character
                fileHandle.closeFile()
                logger.info("Data point saved to \(fileURL.path)")
            } else {
                // Handle error opening the file
                logger.error("Failed to open file for writing.")
            }
        } else {
            // Handle error converting JSON data to string
            logger.error("Failed to convert JSON data to string.")
        }
    } catch {
        // Handle JSON serialization error
        logger.error("JSON serialization error: \(error.localizedDescription)")
    }
}

func deepCopyLast(_ array: [[String: Any]], count n: Int) -> [[String: Any]]? {
    let suffixArray = Array(array.suffix(n))  // Get last `n` elements
    
    do {
        let data = try JSONSerialization.data(withJSONObject: suffixArray, options: [])
        let deepCopy = try JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]]
        return deepCopy
    } catch {
        print("Failed to deep copy last \(n) elements: \(error)")
        return nil
    }
}


@MainActor func getBatteryInfo() -> [String: String] {
    UIDevice.current.isBatteryMonitoringEnabled = true
    let batteryState: String = switch UIDevice.current.batteryState {
        case .charging : "charging"
        case .full : "full"
        case .unplugged : "unplugged"
        default: "unknown"
    }
    return ["batteryLevel": UIDevice.current.batteryLevel.description, "batteryState": batteryState]
}

func copyVideoToDocumentsIfNeeded(filename: String) {
    let fileManager = FileManager.default

    guard let bundleURL = Bundle.main.url(forResource: filename, withExtension: "mov"),
          let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
        logger.error("Could not find video or documents path.")
        return
    }

    let destURL = documentsURL.appendingPathComponent("\(filename).mov")

    if !fileManager.fileExists(atPath: destURL.path) {
        do {
            try fileManager.copyItem(at: bundleURL, to: destURL)
            logger.info("Copied video to: \(destURL)")
        } catch {
            logger.error("Error copying video: \(error)")
        }
    } else {
        logger.warning("Video already exists in Documents.")
    }
}

func createDummyFile(sizeInByte: Int) -> Data{
    let sizeInBytes = sizeInByte
    let data = Data(count: sizeInBytes)
    
    return data
}

func meanAndStd(of values: [Double]) -> (mean: Double, std: Double)? {
    guard !values.isEmpty else { return nil }

    let count = Double(values.count)
    let mean = values.reduce(0, +) / count

    let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / count
    let std = sqrt(variance)

    return (mean, std)
}

func median(of values: [Double]) -> Double? {
    guard !values.isEmpty else { return nil }

    let sorted = values.sorted()
    let count = sorted.count

    if count % 2 == 1 {
        // Odd count — return middle element
        return sorted[count / 2]
    } else {
        // Even count — average the two middle elements
        let mid1 = sorted[(count / 2) - 1]
        let mid2 = sorted[count / 2]
        return (mid1 + mid2) / 2
    }
}

func addPrefix(to filename: String, ext: String) -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd_HHmmss"  // Example: 20250626_154501
    let timestamp = formatter.string(from: Date())

    return "\(filename)_\(timestamp).\(ext)"
}

func getDeviceInfo(device: WAPairedDevice?) -> String {
    return device?.displayName ?? "unknown device"
}

func matchWAParameterTuple(stringParamTuple: (String, String)) -> (WAPerformanceMode, WAAccessCategory) {
    let (modeString, categoryString) = stringParamTuple
    let mode = WAPerformanceMode.allCases.first { "\($0)" == modeString } ?? .realtime
    let category = WAAccessCategory.allCases.first { "\($0)" == categoryString } ?? .bestEffort
    return (mode, category)
}
