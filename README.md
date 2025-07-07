<p align="center"> <img src="doc/img/logo_asl.png" alt="Logo ASL" width="180"> </p>

# Reconocimiento de Lenguaje de Signos ASL mediante IA: Implementación y Comparativa de Rendimiento en Raspberry Pi.

> Autor: **Hugo Gómez Martín** • Tutor: Fernando Rivas Navazo • Co‑tutor: Daniel Urda Muñoz
 
## 📝 Resumen

Este proyecto presenta el desarrollo de un sistema de reconocimiento de lenguaje de signos Americano (ASL) que se basa en visión por computador y aprendizaje profundo, desplegado en una Raspberry Pi 5.

Para llevarlo a cabo, combinamos la implementación de la detección de manos (cvzone) y la clasificación de gestos con una red neuronal convolucional (CNN) entrenada con TensorFlow/Keras sobre un dataset propio de 26 clases con 1000 imágenes cada una. 

El modelo se prepara para su ejecución embebida con TensorFlow Lite y se integra en una aplicación Streamlit que permite 

* **Reconocimiento en tiempo real** desde la cámara, con etiqueta, confianza (%) y *bounding box* con su respectiva Monitorización de **FPS, CPU, RAM y temperatura** de la Raspberry Pi.
* **Clasificación de imágenes estáticas** cargadas por el usuario.
* **Entrenamiento asistido** de nuevos modelos (hasta 26 clases) con descarga automática de un comprimido que contiene el modelo, su optimización y el fichero de texto con sus respectivas etiquetas.

---

## ⚙️ Instalación y puesta en marcha (Windows y Linux)
> Requiere **Python 3.11** o superior. Se recomienda usar un entorno virtual.

### Atajos (Ejecuta estos para evitar comandos (Requieren Python 3.11 o superior))
* **Windows**: `run_app.bat` o `ASL_Detection.exe` en el apartado de **Release**
* **Linux**  : `run_app.sh` (antes `chmod +x run_app.sh` para conceder permisos de ejecución)

#### 1. Clonar el repositorio
`$ git clone https://github.com/hgomezmartin/HandSignDetectionProject.git`
`$ cd HandSignDetectionProject`

#### 2. (Opcional) Crear entorno virtual
`$ python -m venv .venv`
##### Windows
`$ .venv/Scripts/activate`
##### Linux / macOS
`$ source .venv/bin/activate`

#### 3. Instalar dependencias
`$ pip install --upgrade pip`
`$ pip install -r requirements.txt`

#### 4. Instalar el paquete del proyecto
`$ python -m pip install-e .`

#### 5. Lanzar la aplicación
`$ streamlit run src/handsign_asl_detection/web/app.py`

---

## 🚀 Uso básico de la aplicación

1. Selecciona una de las tres secciones desde la barra lateral: **Tiempo Real**, **Foto** o **Entrenamiento**.
2. En **Tiempo Real** pulsa *Iniciar* para ver la predicción sobre la cámara y las métricas de hardware.
3. En **Foto** carga una imagen `.jpg`/`.png`; la inferencia y la confianza se muestran a la derecha (hay imagenes de ejemplo en data/ejemplos).
4. En **Entrenamiento** sube tus carpetas por clase (hasta 26), ajusta *epochs* / *learning rate* / *dropout* y pulsa *Entrenar*.

---

## 🌳 Árbol del proyecto

```
HandSignDetectionProject/
├── .idea/                       
│   └── ...                             # Configuración y metadatos de proyecto
├── .streamlit/
│   └── config.toml                     # Ajustes de Streamlit 
├── data/                       
│   ├── ordered/                        # Dataset ordenado
│   ├── ejemplos/                       # Ejemplos para probar en photo.py
│   └── disordered/                     # Dataset desordenado (usado para entrenar el modelo)
│                
├── doc/
│   ├── anexos.pdf                      # Anexos 
│   └── momoria.pdf                     # Memoria del proyecto (TFG)
├── models/  
│   ├── final_model/                    # Modelo Final Entrenado
│   ├── random_search/                  # Modelo resultante del RS   
│   └── teachable_machine/              # Modelo de TM, donde usamos los .tflite para la app            
├── src/
│   └── handsign_asl_detection/
│       ├── classifier/                 # Lógica de inferencia del modelo
│       │   ├── classifier.py           # Funciones de clasificación 
│       │   ├── classifier_photo.py     # Clasifica imagen estática
│       │   └── classifier_rpi.py       # Adaptaciones para Raspberry Pi
│       ├── config.py                   # Constantes y rutas globales de la aplicación
│       ├── model_creation/             # Código para definir y entrenar tu CNN
│       │   ├── random_search.py        # RS realizado para la busqueda de mejores hiperparámetros
│       │   └── cnn_build.py            # Arquitectura del modelo y funciones de entrenamiento
│       │
│       └── web/                        # Interfaz de usuario con Streamlit
│           ├── app.py                  # Punto de entrada de la APP
│           ├── run.py                  # Para crear el ejecutable
│           ├── components/             # Componentes reutilizables de la UI
│           │   ├── photo.py            # Vista y lógica de “Foto”
│           │   ├── realtime.py         # Vista y lógica de “Tiempo real”
│           │   └── trainer.py          # Vista y lógica de “Entrenamiento”
│           │
│           └── static/                 # Recursos estáticos de la web
│               ├── style.css           # Estilo CSS 
│               └── images/
│                   └── logo_ASL.png    # Logo usado en la UI
├── .gitattributes              
├── .gitignore                  
├── README.md                  
├── pyproject.toml                      # Configuración de build 
├── requirements.txt                    # Dependencias del proyecto 
├── run_app.bat                         # Script Windows para lanzar la app 
└── run_app.sh                          # Script Linux/macOS para lanzar la app 

```

---

## 📜 Créditos y licencias

El proyecto se distribuye bajo licencia **MIT**.  Las principales dependencias usan licencias libres compatibles (Apache 2.0, BSD‑3, MIT…).  Consulta `LICENSE` para más detalles.