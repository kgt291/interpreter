#include "bbi.h"
#include "bbi_prot.h"

int main(int argc, char *argv[])
{
	if (argc == 1) { cout << "¿ë¹ý: bbi filename\n"; exit(1); }
	convert_to_internalCode(argv[1]);
	syntaxChk();
	execute();
	return 0;
}