# Lagerfehlererkennung mit LSTM Neural Networks

Ein Machine Learning Projekt zur automatisierten Erkennung von Lagerfehlern basierend auf Schwingungsdaten unter Verwendung von Long Short-Term Memory (LSTM) Neuronalen Netzwerken.

## Übersicht

Dieses Projekt implementiert ein Deep Learning-System zur frühzeitigen Erkennung von Lagerschäden in rotierenden Maschinen. Durch die Analyse von Schwingungsmustern können potenzielle Ausfälle vorhergesagt werden, was präventive Wartung ermöglicht und ungeplante Stillstände verhindert.

## Technologie-Stack

- **Framework**: PyTorch
- **Neural Network**: LSTM (Long Short-Term Memory)
- **Datenanalyse**: Zeitreihenanalyse von Schwingungsdaten
- **Sprache**: Python

## Dataset

Der verwendete Datensatz stammt aus dem NASA Ames Prognostics Data Repository und enthält hochwertige Schwingungsdaten von Lagern unter verschiedenen Betriebsbedingungen.

**Quelle**: [NASA Ames Prognostics Data Repository](http://ti.arc.nasa.gov/project/prognostic-data-repository)

### Dataset-Charakteristika
- Schwingungssensordaten von rotierenden Lagern
- Verschiedene Fehlermodi und Schadensstufen
- Zeitreihen-basierte Aufzeichnungen
- Realistische Betriebsbedingungen

## Projektziele

- **Frühzeitige Fehlererkennung**: Identifikation von Lagerschäden bevor kritische Ausfälle auftreten
- **Mustererkennung**: Erlernung komplexer Schwingungsmuster mittels LSTM-Netzwerken
- **Präventive Wartung**: Unterstützung wartungsbasierter Entscheidungen
- **Hochpräzise Klassifikation**: Zuverlässige Unterscheidung zwischen gesunden und defekten Lagern

## Projektstruktur

```
Machine-Learning-NeuronaleNetze/
├── Readme.md                 # Projektdokumentation
├── data/                     # Datensätze (lokal zu ergänzen)
├── src/                      # Quellcode
└── results/                  # Ergebnisse und Visualisierungen
```

## Installation & Setup

```bash
# Repository klonen
git clone <repository-url>
cd Machine-Learning-NeuronaleNetze

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install torch torchvision
pip install numpy pandas matplotlib scikit-learn
pip install jupyter notebook
```

## Workflow

1. **Datenvorverarbeitung**: Normalisierung und Segmentierung der Schwingungsdaten
2. **Feature Engineering**: Extraktion relevanter Merkmale aus Zeitreihensignalen
3. **LSTM-Modellierung**: Training rekurrenter Netzwerke für Sequenzklassifikation
4. **Evaluation**: Bewertung der Modellleistung auf Testdaten
5. **Inferenz**: Anwendung auf neue Schwingungsdaten

## Ergebnisse

- Erfolgreiche Klassifikation verschiedener Lagerfehlermodi
- Hohe Genauigkeit bei der Früherkennung von Schäden
- Robuste Performance unter verschiedenen Betriebsbedingungen


## Lizenz

Dieses Projekt steht unter [MIT License](LICENSE).
