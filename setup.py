import subprocess
import sys
import os

def run_command(command):
    """Ejecuta un comando y muestra su salida en tiempo real"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    
    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()
        
        if output == '' and error == '' and process.poll() is not None:
            break
            
        if output:
            print(output.strip())
        if error:
            print(error.strip(), file=sys.stderr)
    
    return process.poll()

def main():
    print("🚀 Iniciando la instalación del entorno...")
    
    # 1. Actualizar pip, setuptools y wheel
    print("\n📦 Actualizando pip, setuptools y wheel...")
    run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")
    
    # 2. Instalar todas las dependencias desde requirements.txt
    print("\n📚 Instalando dependencias desde requirements.txt...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    # 3. Descargar el modelo de SpaCy
    print("\n🔧 Descargando el modelo de SpaCy en inglés...")
    run_command(f"{sys.executable} -m spacy download en_core_web_sm")
    
    print("\n✅ Instalación completada!")
    print("\nPara ejecutar la aplicación, usa el comando:")
    print("python -m streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 