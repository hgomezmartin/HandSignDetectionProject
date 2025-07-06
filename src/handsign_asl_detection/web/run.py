"""
run.py
-------
Script necesario para el ejecutable

Autor: Hugo Gómez Martín
Contacto: hgm1001@alu.ubu.es
Fecha: 05/07/2025
"""
import os.path
import subprocess


def run_streamlit():
    # os.environ["STREAMLIT_THEME_PRIMARY_COLOR"] = "#C51D4A"
    # Obtiene la ruta del script
    script_path = os.path.join(os.path.dirname(__file__), "app.py")

    # Ejecuta la aplicación Streamlit
    subprocess.run(
        ["streamlit", "run", str(script_path),
         "--theme.primaryColor", "#C51D4A",
         "--theme.base", "dark"]
    )


if __name__ == "__main__":
    run_streamlit()
