/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
The network configuration constants.
*/

import WiFiAware
import Network

let appPerformanceMode: WAPerformanceMode = .bulk

let appAccessCategory: WAAccessCategory = .bestEffort
let appServiceClass: NWParameters.ServiceClass = appAccessCategory.serviceClass
