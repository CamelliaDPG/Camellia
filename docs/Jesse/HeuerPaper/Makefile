BASENAME = paper

all: pdf

pdf::
	rubber -W all --pdf ${BASENAME}

ps::
	rubber -W all --ps ${BASENAME}

clean::
	rubber -W all --ps --clean ${BASENAME}
	rubber -W all --pdf --clean ${BASENAME}

