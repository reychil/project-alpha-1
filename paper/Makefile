TITLE="report"
APP_OUTLIERS = "appendix_outliers"
APP_CONV = "appendix_convolution"
APP_SMOOTH = "appendix_smoothing"
APP_CLUSTER = "appendix_clustering"
APP_TIME = "appendix_time"

all:
	pdflatex $(TITLE).tex
	bibtex $(TITLE)
	pdflatex $(TITLE).tex
	pdflatex $(TITLE).tex
	
app:
	pdflatex appendix/$(APP_OUTLIERS).tex
	
	pdflatex appendix/$(APP_CONV).tex
	bibtex $(APP_CONV)
	pdflatex appendix/$(APP_CONV).tex
	pdflatex appendix/$(APP_CONV).tex
	
	pdflatex appendix/$(APP_SMOOTH).tex
	
	# Section not yet written.
	#pdflatex appendix/$(APP_CLUSTER).tex
	#bibtex $(APP_CLUSTER)
	#pdflatex appendix/$(APP_CLUSTER).tex
	#pdflatex appendix/$(APP_CLUSTER).tex
	
	pdflatex appendix/$(APP_TIME).tex
	bibtex $(APP_TIME)
	pdflatex appendix/$(APP_TIME).tex
	pdflatex appendix/$(APP_TIME).tex
	
clean:
	find . -name "*.aux" -o -name "*.log" -o -name "*.bbl" -o -name "*.lof" -o -name "*.lot" -o -name "*.blg" -o -name "*.out" | xargs rm -f

reset: 
	find . -name "*.aux" -o -name "*.log" -o -name "*.bbl" -o -name "*.lof" -o -name "*.lot" -o -name "*.blg" -o -name "*.out" -o -name "*.pdf"  | xargs rm -f
