all:
	make pdf
	make tex
pdf:
	pandoc notes.md -s --template templates/template.tex -V links-as-notes -o notes.pdf
tex:
	pandoc notes.md -s --template templates/template.tex -V links-as-notes -o notes.tex
clean:
	if [ -a notes.pdf ]; then rm notes.pdf; fi;
	if [ -a notes.tex ]; then rm notes.tex; fi;
