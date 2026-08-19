# Marés Oceânicas — documento de estudo
#   make        compila o PDF (três passes, resolve referências cruzadas)
#   make clean  remove os auxiliares
#   make purge  remove também o PDF

DOC = mares-oceanicas

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex
	@echo "--- avisos relevantes ---"
	@awk '/^!/ || /Overfull/ || /undefined/' $(DOC).log || true

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).lof $(DOC).lot

purge: clean
	rm -f $(DOC).pdf

.PHONY: all clean purge
