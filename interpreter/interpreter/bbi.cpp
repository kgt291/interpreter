#include "bbi.h"
#include "bbi_prot.h"
/*만들면서 배우는 인터프리터 : 컴파일러 이론으로 만드는 나만의 스크립트 언어(하야시 하루히코 지음)*/
int main(int argc, char *argv[])
{
	if (argc == 1) { cout << "¿ë¹ý: bbi filename\n"; exit(1); }
	convert_to_internalCode(argv[1]);
	syntaxChk();
	execute();
	return 0;
}
