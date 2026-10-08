import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    deslizador = mo.ui.slider(1, 10, label="Número")
    deslizador
    return (deslizador,)


@app.cell
def _(deslizador, mo):
    mo.md(f"El doble es **{deslizador.value * 2}**")
    return


if __name__ == "__main__":
    app.run()
