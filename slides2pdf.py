import subprocess
import sys


def run_decktape(fl):
    cmd = [
        "decktape",
        "reveal",
        f"_site/slides/{fl}.html",
        f"_site/slides/{fl}.pdf",
    ]
    try:
        subprocess.run(cmd, check=True)
        print("PDF generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    for fl in sys.argv[1:]:
        cmd = [
            "quarto",
            "render",
            f"slides/{fl}.qmd"]
        
        subprocess.run(cmd, check=True)
        
        run_decktape(fl)

