default:
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -rf *~ *.out *.toc *.log *.aux *.blg *.bbl *.bcf *.lol *.run.xml \
	*.synctex.gz *.tdo *-converted-to-* *fdb_latexmk *fls *maf *mtc0 *.mp *.1 \
	*.t1
	find -name "*.aux" -type f -delete

clean_pdf: clean
	@rm -f main.pdf
