		README for utdiss2-05, 25 August 2002


	INTRODUCTION

This is the "new" version of the LaTeX style file package for writing
dissertations at the University of Texas at Austin.

The latest version of this document/package can be obtained from
http://www.ph.utexas.edu/~laser/craigs_stuff/LaTeX/.

I have received my Ph.D. and have accepted a position with the Los
Alamos National Laboratory, to which I will be moving in September.
As much as I hate to say it, I will no longer be able to help with
LaTeX questions from the UT dissertation community.

I will transfer this page to the Office of Graduate Studies shortly.
The new URL isn't defined yet, but I will place a ``redirect'' at
the URL listed above to send your browser to the correct location
when the transition occurs.

If your installation of LaTeX is missing any style files used in this
document (most likely with a \usepackage{package-name.sty} command at
the beginning of disstemplate.tex), take a look at the link on this
page to ``Frequently Requested Style Files'' or on the Comprehensive
TeX Archive Network, \url{http://www.ctan.org}.

I have had the output reviewed by Lynn Renegar, the Doctoral Degree
Evaluator, aka. "The Ruler Lady." She said this template does produce
an acceptable dissertation and an acceptable PDF file.

I have also had the output reviewed by Mike Feissli, the Master's
Degree Plan Evaluator. He said this template does produce acceptable
theses and reports except for the page order (which is different than
that of a dissertation). IF YOU ARE WRITING A MASTER'S THESIS OR REPORT
SEE HIS REVIEW COMMENTS BELOW. MASTER'S DOCUMENTS THAT USE THE DOCTORAL
PAGE ORDER WILL BE REJECTED.

			==================

This is version 5 of the style package file. It addresses:

	- Proper inclusion of only one appendix.
	- An illustration of additional, non-tabular text after
	  a table (the German declension table).

This is version 4 of the style package file addressed:

	- Proper reading from the user's main .tex file of non-default
	  values for \degree, \degreeabbr, and \previousdegrees for
	  master's documents.

Version 3 of the style package file addressed:

	- Proper numbering and formatting of sub-sub-sections on both
	  the sub-sub-section's page and in the table of contents.
	- Proper formatting of pages for master's theses and reports.
	- The difference in page order between the doctoral dissertation
	  (the document default) and master's theses and reports.

Version 2 of the style package file addressed:

	- APA style referencing,
	- Top and bottom margin problems,
	- Vita page number problems in index,
	- Problems with placement of page number on second page
	  (and most likely following pages) of index.

See the section, "ADDITIONAL COMMENTS," below for comments on these points.



	DOCTORAL REVIEW COMMENTS

Lynn Renegar's comments [and mine]:
	- The formatting requirements are based on:

		Turabian, Kate L. / A manual for writers of term papers,
		theses, and dissertations. / Chicago

		LB 2369 T8	PCL Stacks
		LB 2369 T8	Undergraduate Library
		(and other places on campus)

		If you follow the Office of Graduate Studies' guidelines,
		however, there is no need to refer to Kate Turabian's book.

	- There must be NO blank pages in a dissertation [this template
		produces none].
	- A fly page is not needed for the electronic document (it's only
		intended to protect the copyright page when the printed
		version is being bound)[this template produces none].
	- Eventhough the official OGS instructions show the word,
		"certifies," with a capital "C", proper English usage
		should have it as lower-case. [This template produces
		dissertations with a lower-case "C".]
	- TWO supervisors, MAXIMUM [eventhough this template can properly
		handle three].
	- SIX committee members, including supervisor(s), MAXIMUM
		[eventhough this template can properly handle seven].
	- There must be NO titles (professor or Ph.D., etc.) for your
		supervisor or committee members [this template adds none].
	- The following pages are optional:
		- copyright page
		- dedication page
		- acknowledgements page
		- abstract page
	    [I have not checked to see if this template produces the
		correct page numbering if these are left out.]
	- For multiple appendices [as the example document of this template
		has], the lines in the table of contents for each individual
		appendix [e.g., the one in the example document that says,
			Appendix A. Lerma's Appendix		29  ]
		should be indented from the left margin under the line
		that says, "Appendices." [e.g., the one in the example
		document that says,
			Appendices				28  ]
		[This template produces lines flush with the left margin,
		but that is acceptable to her now.]
	- Use the quote environment, not the quotation environment,
		for quotes more than four lines long. Line spacing in
		the block quote of space-and-a-half is acceptable.
	- An index is acceptable and should be after the bibliography
		but before the vita [as the one in the example document is].
	- Your bibliography should at least include the materials cited in
		the text of your disseration; it may also include
		references not cited.


	MASTER'S REVIEW COMMENTS

Mike Feissli's comments [and mine]:

	- The proper page order for master's documents is
		- copyright
		- title page
		- signature page
		- <remainder of document>
		[For a dissertation, the Committee Certification
		(signature) page is before the title page.]
	- For master's documents nothing is permitted to come between
		the bibliography and the vita; if you have an index
		it must be before the bibliography.
		[For a dissertation, the index must be after the
		bibliography (like a book) but before the vita (which
		isn't usually part of a book). The placement of the
		index in the formatted document is controlled by the
		placement of the one line command \printindex% in the
		.tex file. Its placement is now well commented in
		disstemplate.tex.]


	ADDITIONAL COMMENTS

1.  In the first version of this README, I said,

	I inferred from Lynn Renegar's comments that she finds
	the APA style unacceptable and have removed it from this
	template (eventhough Miguel Lerma had it commented out
	at the end of his .sty file).

    I have since found out that this statement was NOT correct. The only
    thing to which Lynn Renegar objected was the typical APA style of not
    putting the page number 1 on the first page. I received comments from
    a friend who did try out APA style referencing and have put them in a
    separate file, README.APA. Please look in that file for suggestions if
    you want to use APA style referencing.

2.  I have made all the chapters and appendicies separate files which
    are referenced in disstemplate.tex by include statements such as,

		\include{chapter-introduction}

    This makes the disstemplate.tex file shorter and easier to modify
    to be your own dissertation.

3.  I have not checked to see that this template correctly handles
    the changes needed for a Doctor of Musical Arts. I know "Dissertation"
    needs to be changed to "Treatise" and there are some changes on the
    committee certification page I don't recall. I believe this template
    does not correctly handle all of the changes needed. If you run into
    difficulty, see Lynn Renegar (232-3630). I _might_ be able to help out
    during the Spring of 2002.

4.  This template does not explicitly produce:

	- a glossary,
	- a List of Illustrations,
	- a Nomenclature page, or,
	- a List of Supplemental Files.

5.  I have found that the version of the software used to generate
    the PostScript and way you print the PostScript file effects
    the size of the top and bottom margins. The document I printed
    for Lynn Renegar's review obviously met the Office of Graduate
    Studies requirements (except for one page, see 7 below). Since
    then I have compiled my document on a different operating system
    and have found that it no longer meets OGS requirements.

    To solve this problem I have moved the command that defines
    the top margin from the style file, utdiss2-nn.sty, to
    disstemplate.tex. There are instructions in disstemplate.tex
    for changing the value of \topmargin to what you need.

6.  The page number on the second and following pages of the index is
    in the wrong place (it's lower than it should be). It was in the
    wrong place on the officially reviewed copy but was not caught (in
    a microfilmed version, the page number would not appear on the film).
    I specifically asked Lynn Renegar about this and she said she was
    not concerned. I did NOT ask Mike Feissli about this.

    I do not yet know the cause of this problem nor its solution.

7.  If listed in the index, the vita page will be listed with the
    wrong page number, specifically the first page of the index.
    Apparently the commands which make up the index expect the
    index to be the last thing in the document. We, however, must
    put the Vita last. The fix to this is to not list the Vita
    or anything else in the Vita in the index.



	COMPILING YOUR DOCUMENT

First copy the template file, disstemplate.tex, to another file name
with the .tex suffix (e.g., mydiss.tex). Modify the new file to make it
your dissertation. (It's always best to leave a copy of the original on
your system so you can refer to it later.) To do this on a Linux or
other Unix system, execute the following commands,

		cp disstemplate.tex mydiss.tex
		chmod 644 mydiss.tex

The file Makediss is a shell script (program) for running on Linux or
other Unix systems. You will need to edit it and enter the name of your
dissertation file before you run it. On the line that begins,

		DISS="./disstemplate"

replace the word disstemplate with the name of your file (without the
.tex suffix).

If you are using a different operating system, the following is the list
of commands you will have to run to properly create your
dissertation's PostScript file:


	Copy the file disstemplate.tex to another file name with the .tex
		suffix (e.g., mydiss.tex). Modify the new file to make it
		your dissertation. (It's always best to leave a copy of
		the original on your system so you can refer to it later.)

	Copy the current version of utdiss2-nn.sty to utdiss2.sty.

	Execute the following commands,

	latex mydiss		(latex generates mydiss.aux)
	bibtex mydiss		(bibtex generates mydiss.bbl)
	latex mydiss		(latex generates new mydiss.aux)
	latex mydiss		(latex resolves cross references)
	makeindex mydiss	(makeindex reads mydiss.idx and
				   generates mydiss.ind with the index
				   entries sorted)
	latex mydiss		(latex includes the Index)
	latex mydiss		(latex generates the final version of
				   mydiss.dvi)


You will now have your dissertation in a .dvi file. You have two choices,

	1) To print from the .dvi file, execute,

	dvips mydiss

	2) To make the postscript file from the .dvi file

	dvips mydiss -o

The final thing that needs to be done for the OGS is to convert the
PostScript file into a single Adobe PDF file. I have not yet explored this
problem. Lynn Renegar said that some approaches to coversion caused the page
numbers to be incorrect. I cannot clarify that situation yet.



	GOODBYE AND GOOD LUCK

I have re-written the example document (disstemplate.tex) to be more
explicit in its comments about what has to be done, hopefully that will
be sufficient to get you on the road with your own dissertation. If you
have any questions, however, email me at mccluskey@mail.utexas.edu.
I cannot promise I will be able to help you immediately or very much
since I also am working on a dissertation of my own during the Spring 2002
semester.

					Craig McCluskey
					15 March 2002
