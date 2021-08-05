LATEX=pdflatex
MKF=.:/usr/local/texlive/texmf-local/fonts//:/usr/local/texlive/2020/texmf-dist/tex/latex/apacite/apacite.sty:/usr/local/texlive/2020/texmf-var/fonts//:/usr/local/texlive/2020/texmf-dist/tex/context/fonts//:/usr/local/texlive/2020/texmf-dist/tex/csplain/fonts//:/usr/local/texlive/2020/texmf-dist/tex/csplain/fonts//:/usr/local/texlive/2020/texmf-dist/doc/context/sources/general/manuals/fonts//:/usr/local/texlive/2020/texmf-dist/doc/fonts//:/usr/local/texlive/2020/texmf-dist/doc/plain/apprendre-a-programmer-en-tex/fonts//:/usr/local/texlive/2020/texmf-dist/doc/support/ketcindy/ketcindyjs/katex/fonts//:/usr/local/texlive/2020/texmf-dist/fonts//:/usr/local/texlive/2020/texmf-dist/metapost/context/fonts//:/usr/local/texlive/2020/texmf-dist/source/latex/lilyglyphs/fonts//:/usr/local/texlive/2020/texmf-dist/source/fonts//

all: clean msc

msc:
	#TFMFONTS=$(MKF) $(LATEX) document-msc
	makeindex -s authorindex.ist  -o document.and document.adx
	makeindex -s nomenclature.ist -o document.nls document.nlo
	makeindex document-msc
	TFMFONTS=$(MKF) $(LATEX) document-msc
	bibtex document-msc
	#TFMFONTS=$(MKF) $(LATEX) document-msc
	#TFMFONTS=$(MKF) $(LATEX) document-msc
	#TTFONTS=$(MKF) TFMFONTS=$(MKF) dvips document
	#psnup -2 document.ps | pstops "4:-3L@.7(21cm,0)+0L@.7(21cm,14.85cm),1L@.7(21cm,0)+-2L@.7(21cm,14.85cm)" > a2.ps

clean:
	rm *.bbl *.aux document-msc.pdf