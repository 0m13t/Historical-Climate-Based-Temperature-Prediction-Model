Hier is de Engelse versie, met dezelfde structuur en inhoud:

**Introduction:**
After taking several courses and with the start of my first internship approaching, I wanted to test my knowledge of data science by building a machine learning model. I chose a concrete question: can I predict the temperature for a future holiday destination?

For this project, I used the following tools: Python 3.12.4, XGBoost, the Open-Meteo Geocoding API, and the NAO index from NOAA’s Climate Prediction Center. I also used Gemini Flash 3.5 as a support tool to test ideas, improve code, and critically reflect on technical choices.

**Approach:**
The code was developed through multiple iterations. In the first version, the model directly predicted the maximum temperature based on date features, air pressure, and climate indices such as NAO and ONI. This approach already showed that there were predictive patterns in the data, but it also made clear that data quality and time alignment were important.

In the second version, I improved the data alignment to prevent NaN values in the dataset. The weather data and climate indices were combined more strictly by year and month, so that only periods with all required data available were used. This made the dataset more stable and prevented the model from training on incomplete rows. This was also the stage where the first temperature predictions were tested, resulting in an MAE of 3.05°C.

After that, I looked more critically at how weights were distributed between years. Because of climate change, I decided to give older weather data a lower sample weight. For this, I tested the following strategies:

| Strategy            |    MAE | Interpretation                                     |
| ------------------- | -----: | -------------------------------------------------- |
| Quadratic weighting | 2.75°C | Best performing                                    |
| Linear weighting    | 2.88°C | Stable and currently used in the main code         |
| Exponential decay   | 3.10°C | Too aggressive; older data lost too much influence |
| Uniform weighting   | 3.30°C | Treated everything equally                         |

Although quadratic weighting achieved the lowest MAE in this experiment, I chose to keep linear weighting in the main version of the model because it is a more conservative and stable approach. Linear weighting still gives more importance to recent years, which helps the model account for climate change, but it does not reduce the influence of older data too aggressively.

**Evaluation:**
To test the model, I used a chronological split between years. To evaluate how well the model could be applied to future data, I chose 2024 and 2025 as the test years.

**Critical Reflection:**
The main improvement I see is moving the model from daily prediction to period-based prediction. A prediction over a week or month would likely improve reliability, because daily temperature fluctuates more strongly than averages over a longer period.

The model was mainly tested on Baarn. In tests with Toulon, where an MAE of 1.73°C was achieved, the model performed much better. This shows that the model does not yet perform consistently across different locations.
