**Voorwoord:**
Na een aantal colleges en met de start van mijn eerste stage in zicht wilde ik mijn kennis op het gebied van data science testen door een ML-model te bouwen. Ik koos voor een concreet vraagstuk: kan ik de temperatuur voorspellen voor een toekomstige vakantiebestemming?

Voor dit project heb ik gekozen voor de volgende tools: Python 3.12.4, XGBoost, Open-Meteo Geocoding API en de NAO-index van NOAA’s Climate Prediction Center. Daarnaast heb ik Gemini Flash 3.5 als hulpmiddel gebruikt om ideeën te toetsen, code te verbeteren en keuzes kritisch te bekijken.

**Approach:**
De code is door meerdere iteraties heen ontwikkeld. In de eerste versie werd de maximumtemperatuur direct voorspeld op basis van datumfeatures, luchtdruk en klimaatindices zoals NAO en ONI. Deze aanpak liet al zien dat er voorspellende patronen in de data aanwezig waren, maar maakte ook duidelijk dat datakwaliteit en tijdsuitlijning belangrijk waren.

In de tweede versie heb ik daarom de data-alignment verbeterd om NaN-waarden in de dataset te voorkomen. De weerdata en klimaatindices werden strikter op jaar en maand gecombineerd, zodat alleen periodes werden gebruikt waarvoor alle benodigde data beschikbaar was. Dit maakte de dataset stabieler en voorkwam dat het model trainde op incomplete rijen. Dit was ook het moment waarop de eerste temperatuurvoorspellingen werden getest, waarbij een MAE van 3.05°C werd geregistreerd.

Daarna ben ik kritischer gaan kijken naar de verdeling van weights tussen de jaren. Vanwege klimaatverandering heb ik besloten om oudere weerdata een lagere sample weighting toe te kennen. Hiervoor heb ik de volgende strategieën getest:

| Strategy            |   MAE | Interpretation                                     |
| ------------------- | ----- | -------------------------------------------------- |
| Quadratic weighting | 2.75C | Best performing                                    |
| Linear weighting    | 2.88C | Stable and currently used in the main code         |
| Exponential decay   | 3.10C | Too aggressive; older data lost too much influence |
| Uniform weighting   | 3.30C | Treated everything equally                         |

Although quadratic weighting achieved the lowest MAE in this experiment, I chose to keep linear weighting in the main version of the model because it is a more conservative and stable approach. Linear weighting still gives more importance to recent years, which helps the model account for climate change, but it does not reduce the influence of older data too aggressively.

**Evaluation**
Om het model te testen heb ik een chronologische splitsing tussen de jaren gemaakt. Om de toepasbaarheid op toekomstige data te beoordelen, heb ik ervoor gekozen om de jaren 2024 en 2025 als testjaren te gebruiken.

**Critical Reflection**
Als grootste verbeterpunt zou ik zeggen dat het model van een dagvoorspelling naar een periodevoorspelling moet gaan. Een voorspelling over een week of maand zou de betrouwbaarheid van de voorspelling waarschijnlijk kunnen vergroten, omdat dagelijkse temperatuur sterker schommelt dan gemiddelden over een langere periode.

Ook is het model vooral getest op Baarn. Uit testen met Toulon, waar een MAE van 1.73°C werd behaald, bleek dat het model op sommige locaties veel beter presteert. Dit laat zien dat het model nog niet consistent genoeg presteert over verschillende locaties.
