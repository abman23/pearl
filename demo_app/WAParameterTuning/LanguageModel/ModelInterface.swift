//
//  ModelInterface.swift
//  WAFileTransmission
//
//  Created by Yanqing Lu on 8/14/25.
//  Copyright © 2025 Apple. All rights reserved.
//
import FoundationModels
import OSLog
import Foundation

struct ModelInterface {
    static func modelCompletion(contextsList: [[String: String]]) async -> ((String, String), String) {
        let session = LanguageModelSession()
        let options = GenerationOptions(temperature: 0.1)
        let prompt = formatPrompt(contextsList: contextsList)
//        print("Prompt: \(prompt)")
        var parameterTuple = ("unknownMode", "unknownCategory")
        var text = "AFM no response"
        do {
            // Get response from the model
            text = try await session.respond(to: prompt, options: options).content
//            print("AFM response: \(text)")
            // Post-process to extract parameter tuple
            let perfPattern = #"\b(bulk|realtime)\b"#
            let accessPattern = #"\b(bestEffort|background|interactiveVideo|interactiveVoice)\b"#

            let perfRegex = try! NSRegularExpression(pattern: perfPattern, options: [])
            let accessRegex = try! NSRegularExpression(pattern: accessPattern, options: [])
            
            let range = NSRange(location: 0, length: text.utf16.count)
            
            // Find matches
            let perfMatches = perfRegex.matches(in: text, range: range)
            let accessMatches = accessRegex.matches(in: text, range: range)
            
            if let firstPerf = perfMatches.last,
               let firstAccess = accessMatches.last,
               let perfRange = Range(firstPerf.range, in: text),
               let accessRange = Range(firstAccess.range, in: text) {
                
                let performanceMode = String(text[perfRange])
                let accessCategory = String(text[accessRange])
                parameterTuple = (performanceMode, accessCategory)
            }
            
//            print("Parameter tuple: \(parameterTuple)")
        } catch {
            logger.error("Failed to get response from AFM")
        }
        
        return (parameterTuple, text)
    }
    
    static func formatPrompt(contextsList: [[String: String]], n_past_steps: Int = 10) -> String {
        var contextsLog: String = ""
        for contexts in contextsList.suffix(n_past_steps) {
            let localBatteryLevel: String = contexts["localBattery"] ?? "0.0"
            let remoteBatteryLevel: String = contexts["remoteBattery"] ?? "0.0"
            let time: String = contexts["time"] ?? "unknown time"
            let application: String = contexts["dataType"] ?? "unknown app"
            
            let contextsEntry = "| \(time) | \(application) | \(localBatteryLevel) | \(remoteBatteryLevel) |"
            contextsLog += "\n\(contextsEntry)"
        }
        
        let prompt: String = """
            Task:
            You are a system optimizer responsible for configuring Wi-Fi Aware parameters between two devices (sender and receiver) to optimize performance based on current operating conditions.

            Goal:
            Select the optimal combination of the following two Wi-Fi Aware parameters:

            1. performanceMode: Choose one of:
            - 'bulk': Optimizes energy efficiency but has higher latency.
            - 'realtime': Minimizes latency but increases energy consumption.

            2. accessCategory: Choose one of:
            - 'bestEffort'
            - 'background'
            - 'interactiveVideo'
            - 'interactiveVoice'

            The objective is to jointly optimize latency and energy consumption for both devices, while the effects of context information should be considered. 
            Prioritize latency when the typical applications in the current time period are latency-critical and the battery level is high.
            Prioritize energy consumption when the battery level is low or the applications are not latency-critical.

            Context Log entries:
            - Time of day: In the format of hh:mm:ss. There are typical app usage patterns at certain time of day.
            - Application: The name of the Wi-Fi Aware application at the time. Infer the latency requirement from the name.
            - Battery level: A number between 0 and 1. Lower battery levels increase the weight of energy usage in decision-making.

            Context Log:
            | Time of day | Application | Receiver Battery Level | Sender Battery Level |\(contextsLog)
            
            Instruction:
            Output your selected parameter tuple following the output format below. Do not analyze each context log entry. Only add one concise overall ratioanle for your decision. 

            Output Format:
            (performanceMode, accessCategory)
            """
        
        return prompt
    }
}
