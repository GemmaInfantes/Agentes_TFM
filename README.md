# Sistema de Análisis de Documentos con Agentes Inteligentes

Este sistema analiza documentos utilizando múltiples agentes especializados para extraer información relevante, incluyendo metadatos, resúmenes, palabras clave, temas, estructura e insights.

## Requisitos Previos

1. **Python 3.8 o superior**
   - Descarga Python desde [python.org/downloads](https://www.python.org/downloads/)
   - **IMPORTANTE**: Durante la instalación, marca la casilla "Add Python to PATH"
   - Reinicia tu computadora después de la instalación

2. **Dependencias del Sistema**
   - pip (gestor de paquetes de Python)
   - setuptools
   - wheel

## Instalación y Configuración

1. **Clonar el repositorio**
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd AGENTES--main
   ```

2. **Crear y activar entorno virtual**
   ```bash
   # Crear el entorno virtual
   python -m venv venv

   # Activar el entorno virtual
   # En Windows:
   .\venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Actualizar pip y herramientas básicas**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

4. **Instalar dependencias**
   ```bash
   python -m pip install -r requirements.txt
   ```

5. **Instalar modelo de SpaCy**
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Configurar Azure OpenAI**
   - Asegúrate de tener un archivo `configs/openai_config.py` con tus credenciales de Azure OpenAI
   - El archivo debe contener:
     - endpoint
     - model_name
     - deployment
     - api_version
     - subscription_key

## Ejecución

1. **Activar el entorno virtual** (si no está activado)
   ```bash
   # En Windows:
   .\venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

2. **Iniciar la aplicación**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Acceder a la interfaz**
   - La aplicación se abrirá automáticamente en tu navegador
   - Si no se abre, visita: `http://localhost:8512`

## Uso de la Aplicación

1. **Subir Documentos**
   - Haz clic en "Browse files" o arrastra un archivo
   - Formatos soportados: PDF, DOCX, TXT

2. **Configurar Análisis**
   En la barra lateral, selecciona qué análisis quieres ver:
   - Metadata
   - Resumen
   - Palabras Clave
   - Temas
   - Estructura
   - Insights
   - Vectorización

3. **Iniciar Análisis**
   - Haz clic en "🚀 Iniciar Análisis"
   - Espera a que se complete el procesamiento
   - Los resultados se mostrarán en pestañas separadas

## Estructura del Proyecto

```
AGENTES--main/
├── configs/              # Configuraciones (incluye credenciales)
├── src/                  # Código fuente
├── uploaded_docs/        # Documentos subidos temporalmente
├── venv/                 # Entorno virtual
├── streamlit_app.py      # Aplicación principal
├── requirements.txt      # Dependencias
└── setup.py             # Script de configuración
```

## Solución de Problemas

1. **Error: Python no encontrado**
   - Verifica que Python está instalado: `python --version`
   - Asegúrate de que Python está en el PATH
   - Reinicia la terminal después de instalar Python

2. **Error: Módulos no encontrados**
   - Asegúrate de que el entorno virtual está activado
   - Ejecuta: `python -m pip install -r requirements.txt`
   - Verifica que todas las dependencias se instalaron correctamente

3. **Error: Credenciales de Azure OpenAI**
   - Verifica que el archivo `configs/openai_config.py` existe
   - Comprueba que las credenciales son correctas

4. **Error al activar el entorno virtual**
   - En Windows, si hay problemas con la ejecución de scripts:
     ```bash
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```

## Notas Importantes

- Los documentos subidos se procesan temporalmente y se eliminan después del análisis
- Asegúrate de tener suficiente espacio en disco para procesar documentos grandes
- La aplicación requiere conexión a internet para acceder a Azure OpenAI
- Siempre trabaja dentro del entorno virtual para evitar conflictos de dependencias

## Soporte

Si encuentras algún problema o tienes preguntas, por favor:
1. Revisa la sección de Solución de Problemas
2. Verifica que todos los requisitos están instalados correctamente
3. Asegúrate de que las credenciales de Azure OpenAI son válidas
4. Comprueba que estás trabajando dentro del entorno virtual

## Instalación de Milvus con Docker en Windows

Si tienes Windows y quieres usar los contenedores de Docker:

### Prerrequisitos
Necesitas instalar estos componentes primero:
- Install Docker Desktop
- Install Windows Subsystem for Linux 2 (WSL 2)
- Install Python 3.8+

### Opción 1: Usar PowerShell o Command Prompt

1. Abre Docker Desktop en modo administrador haciendo clic derecho y seleccionando "Run as administrator"

2. Descarga el script de instalación y guárdalo como standalone.bat:
```powershell
Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
```

3. Ejecuta el script para iniciar Milvus:
```powershell
standalone.bat start
```

Espera a que Milvus inicie:
```
Wait for Milvus starting...
Start successfully.
```

Para cambiar la configuración predeterminada de Milvus, edita user.yaml y reinicia el servicio.

### Opción 2: Usar WSL 2

1. Inicia WSL 2:
```powershell
wsl --install
```

2. Descarga el script de instalación:
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```

3. Inicia el contenedor Docker:
```bash
bash standalone_embed.sh start
```

### Después de la instalación

Después de ejecutar el script de instalación:
- Un contenedor Docker llamado `milvus-standalone` se habrá iniciado en el puerto 19530
- Un etcd embebido se instala junto con Milvus en el mismo contenedor y sirve en el puerto 2379
- El volumen de datos de Milvus se mapea a `volumes/milvus` en la carpeta actual

## Uso

1. Inicia la aplicación:
```bash
streamlit run streamlit_app.py
```

2. Abre tu navegador en `http://localhost:8501`

3. Sube un documento PDF o DOCX

4. El sistema procesará el documento y mostrará:
   - Metadatos extraídos
   - Resumen del contenido
   - Palabras clave identificadas
   - Temas principales
   - Estructura del documento
   - Insights generados

## Estructura del Proyecto

```
.
├── src/
│   ├── agent_*.py      # Agentes individuales
│   ├── graph_builder.py # Construcción del grafo de agentes
│   ├── state.py        # Definición del estado
│   └── utils.py        # Utilidades comunes
├── streamlit_app.py    # Aplicación Streamlit
└── requirements.txt    # Dependencias
```

## Contribuir

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles. 