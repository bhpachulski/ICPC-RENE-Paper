main.pdf: main.tex
	# you need to run pdflatex command twice if you're having issues 
	# getting TOC or top progress bar to show up in resulting pdf	
	xelatex main.tex

view-xpdf: main.pdf
	xpdf main.pdf & disown	

view-okular: main.pdf
	okular main.pdf & disown	

view-acroread: main.pdf
	acroread main.pdf & disown	

view-evince: main.pdf
	evince main.pdf & disown	

clean:
	latexmk -c
	rm -rf *run.xml* *.snm* *blx.bib* *.nav*