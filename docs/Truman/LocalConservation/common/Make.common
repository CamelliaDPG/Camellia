vectorsources := $(shell find rawfigs/ ../common/rawfigs/ -name '*.dia' -o -name '*.eps' -o -name '*.m' -o -name '*.pdf' -o -name '*.ps' -o -name '*.py' -o -name '*.svg' 2>/dev/null)
rastersources := $(shell find rawfigs/ ../common/rawfigs/ -name '*.jpg' -o -name '*.gif' 2>/dev/null)
readysources := $(shell find rawfigs/ ../common/rawfigs/ -name '*.png' -o -name '*.pdf' -o -name '*mpg' -o -name '*.mpeg' 2>/dev/null)
texsource := $(wildcard abstract.tex paper.tex poster.tex proposal.tex report.tex talk.tex)

vectorfigs := $(shell echo ' ' $(patsubst %_fm.eps,%.eps,$(vectorsources)) ' ' | sed -e 's> \(../common/\)*raw> >g' -e 's/\.[^. ]* /.pdf /g')
rasterfigs := $(shell echo ' ' $(rastersources) ' ' | sed -e 's> \(../common/\)*raw> >g' -e 's/\.[^. ]* /.png /g')
readyfigs := $(shell echo ' ' $(readysources) ' ' | sed -e 's> \(../common/\)*raw> >g')
figures := $(vectorfigs) $(rasterfigs) $(readyfigs)
dirname  := $(shell basename $(shell pwd))
ifeq ($(words $(texsource)),'1')
    texfinal := $(dirname).pdf
    texroot   = $(patsubst %.pdf,%,$(1))
else
    texfinal := $(patsubst %.tex,$(dirname)-%.pdf,$(texsource))
    texroot   = $(patsubst $(dirname)-%,%,$(patsubst %.pdf,%,$(1)))
endif

bibfiles := $(wildcard *.bib)
styfiles := $(wildcard *.sty ../common/*.sty)
clsfiles := $(wildcard *.cls ../common/*.cls)

BIBTEX     ?= bibtex
PDFLATEX   ?= pdflatex
LATEX      ?= latex
PYTHON     ?= python
OCTAVE     ?= octave
INKSCAPE   ?= inkscape
FRAGMASTER ?= fragmaster

MYPDFLATEX   = TEXINPUTS=$(abspath ../common):$$TEXINPUTS $(PDFLATEX) -halt-on-error -interaction=nonstopmode
MYLATEX      = TEXINPUTS=$(abspath ../common):$$TEXINPUTS $(LATEX) -halt-on-error -interaction=nonstopmode
MYFRAGMASTER = TEXINPUTS=$(abspath ../common):$$TEXINPUTS $(FRAGMASTER)

.PHONY: all figures clean cleanlatex cleanfigs

all: $(texfinal)

figures: $(figures)

$(texfinal): *.tex $(bibfiles) $(figures) $(clsfiles) $(styfiles)
	if ls *.bib 2>&1; then $(MYPDFLATEX) -draftmode $(call texroot,$@); fi
	if (ls *.bib 2>&1 && ls $(call texroot,$@).aux 2>&1); then $(BIBTEX) $(call texroot,$@); fi
	$(MYPDFLATEX) -draftmode $(call texroot,$@) && $(MYPDFLATEX) -draftmode $(call texroot,$@) && $(MYPDFLATEX) $(call texroot,$@)
	if [ "$@" != "$(call texroot,$@).pdf" ]; then mv "$(call texroot,$@).pdf" "$@"; fi

clean: cleanlatex cleanfigs

cleanlatex:
	rm -f $(patsubst %.tex, %.aux, $(wildcard *.tex))
	rm -f $(foreach ext,.bbl .blg .log .dvi .nav .nlo .out .pdf .snm .spl .toc .vrb Notes.bib,$(patsubst %.tex,%$(ext),$(texsource)))
	rm -f $(texfinal)

cleanfigs:
	rm -rf figs/

figs/%.pdf: rawfigs/%.dia
	@mkdir -p $(dir $@)
	dia -t eps-builtin -e $?_roytemp.eps $? && epstopdf $?_roytemp.eps -o=$@
	@rm -f $?_roytemp.eps

figs/%.pdf: rawfigs/%.eps
	@mkdir -p $(dir $@)
	epstopdf $? -o=$@ || (rm $@; exit 1)

figs/%.pdf: rawfigs/%_fm rawfigs/%_fm.eps  # fragmaster(1) with optional control file
	@mkdir -p $(dir $@)
	@(test -f rawfigs/fragmaster.dfm && (cd $(dir $@) && ln -sf -t . ../rawfigs/fragmaster.dfm) || true)
	@(cd $(dir $@) && ln -sf -t . $(addprefix ../,$+) && $(MYFRAGMASTER))

figs/%.pdf: rawfigs/%.m
	@mkdir -p $(dir $@)
	@cd $(dir $?) && $(OCTAVE) $(notdir $?)
	@(test -f $(dir $?)/$*.eps && (epstopdf $(dir $?)/$*.eps -o=$@; rm -f $(dir $?)/$*.eps) || true)
	@(test -f $(dir $?)/$*.pdf && mv $(dir $?)/$*.pdf $@ || true)

figs/%.pdf: rawfigs/%.py
	@mkdir -p $(dir $@)
	@cd $(dir $?) && $(PYTHON) $(notdir $?)
	@(test -f $(dir $?)/$*.eps && (epstopdf $(dir $?)/$*.eps -o=$@; rm -f $(dir $?)/$*.eps) || true)
	@(test -f $(dir $?)/$*.pdf && mv $(dir $?)/$*.pdf $@ || true)

figs/%.pdf: rawfigs/%.ps
	@mkdir -p $(dir $@)
	ps2pdf $? $@

#figs/%.pdf: rawfigs/%.png
#       convert $? $@

figs/%.pdf: rawfigs/%.svg
	@mkdir -p $(dir $@)
	$(INKSCAPE) $? -z --export-pdf=$@

figs/%.png: rawfigs/%.jpg
	@mkdir -p $(dir $@)
	convert $? $@

figs/%.png: ../common/rawfigs/%.jpg
	@mkdir -p $(dir $@)
	convert $? $@

figs/%.mpeg: rawfigs/%.mpeg
	@mkdir -p $(dir $@)
	@reldir=`echo $(dir $@) | sed -e 's>[^/]*/*>../>g'`; ln -sf $${reldir}$? $@

figs/%.mpg: rawfigs/%.mpg
	@mkdir -p $(dir $@)
	@reldir=`echo $(dir $@) | sed -e 's>[^/]*/*>../>g'`; ln -sf $${reldir}$? $@

figs/%.pdf: rawfigs/%.pdf
	@mkdir -p $(dir $@)
	@reldir=`echo $(dir $@) | sed -e 's>[^/]*/*>../>g'`; ln -sf $${reldir}$? $@

figs/%.png: rawfigs/%.png
	@mkdir -p $(dir $@)
	@reldir=`echo $(dir $@) | sed -e 's>[^/]*/*>../>g'`; ln -sf $${reldir}$? $@

figs/%.png: ../common/rawfigs/%.png
	@mkdir -p $(dir $@)
	@reldir=`echo $(dir $@) | sed -e 's>[^/]*/*>../>g'`; ln -sf $${reldir}$? $@

figs/%.pdf: ../common/rawfigs/%.pdf
	@mkdir -p $(dir $@)
	@reldir=`echo $(dir $@) | sed -e 's>[^/]*/*>../>g'`; ln -sf $${reldir}$? $@

figs/%.pdf: ../common/rawfigs/%.eps
	@mkdir -p $(dir $@)
	epstopdf $? -o=$@

figs/%.pdf: ../common/rawfigs/%.ps
	@mkdir -p $(dir $@)
	ps2pdf $? $@

figs/%.pdf: ../common/rawfigs/%.svg
	@mkdir -p $(dir $@)
	$(INKSCAPE) $? -z --export-pdf=$@
