prompt_minimal = "{context_logs}"

prompt_joint = """Task:
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

Inputs:
Sender device: {sender_device}
Receiver device: {receiver_device}
Context Log:
{context_logs}

Decision Output:
(performanceMode, accessCategory)="""

prompt_latency = """Task:
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

The objective is to minimize the latency for the Wi-Fi Aware connection.

Context Log entries:
- Time of day: In the format of hh:mm:ss. There are typical app usage patterns at certain time of day.
- Application: The name of the Wi-Fi Aware application at the time. Infer the latency requirement from the name.

Inputs:
Sender device: {sender_device}
Receiver device: {receiver_device}
Context Log:
{context_logs}

Decision Output:
(performanceMode, accessCategory)="""

prompt_battery = """Task:
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

The objective is to minimize energy consumption for the Wi-Fi Aware connection.

Context Log entries:
- Battery level of sender and receiver: A number between 0 and 1. Lower battery levels increase the weight of energy usage in decision-making.

Inputs:
Sender device: {sender_device}
Receiver device: {receiver_device}
Context Log:
{context_logs}

Decision Output:
(performanceMode, accessCategory)="""
