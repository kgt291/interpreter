#include <iostream>
#include <cstdlib>
#include <cctype>
using namespace std;

enum TknKind {
	Print, Lparen, Rparen, Plus, Minus, Multi, Divi, Assign, VarName, IntNum, EofTkn, Others
};

struct Token {
	TknKind kind;
	int intVal;
	Token() { kind = Others; intVal = 0; }
	Token(TknKind k, int d = 0) { kind = k; intVal = d; }
};

void input();
void statement();
void expression();
void term();
void factor();
Token nextTkn();
int nextCh();
void operate(TknKind op);
void push(int n);
int pop();
void chkTkn(TknKind kd);

#define STK_SIZ 20
int stack[STK_SIZ + 1];
int stkct;
Token token;
char buf[80], *bufp;
int ch;
int var[26];
int errF;

int main() {
	while (1) {
		input();
		token = nextTkn();
		if (token.kind == EofTkn) exit(1);
		statement();
		if(errF) cout << " --err--\n";
	}
	return 0;
}

void input() {
	errF = 0; stkct = 0;
	cin.getline(buf, 80);
	bufp = buf;
	ch = nextCh();
}

void statement() {
	int vNbr;

	switch (token.kind) {
	case VarName:
		vNbr = token.intVal;
		token = nextTkn();
		chkTkn(Assign); if (errF) break;
		token = nextTkn();
		expression();
		var[vNbr] = pop();
		break;
	case Print:
		token = nextTkn();
		expression();
		chkTkn(EofTkn); if (errF) break;
		cout << " " << pop() << endl;
		return;
	default:
		errF = -1;
	}
	chkTkn(EofTkn);
}

void expression() {
	TknKind op;

	term();
	while(token.kind == Plus || token.kind == Minus) {
		op = token.kind();
		token = nextTkn(); term(); operate(op);
	}
}

void term() {
	TknKind op;
	factor();
	while (token.kind == Multi || token.kind == Divi) {
		op = token.kind();
		token = nextTkn(); factor(); operate(op);
	}
}

void factor() {
	switch (token.kind) {
	case VarName:
		push(var[token.intVal]);
		break;
	case IntNum:
		push(token.intVal);
		break;
	case Lparen:
		token = nextTkn();
		expression();
		chkTkn(Rparen);
		break;
	default:
		errF = 1;

	}
	token = nextTkn();
}

Token nextTkn() {
	TknKind kd = Others;
	int num;

	while(isspace(ch)) ch = nextCh();
	if (isdigit(ch)) {
		for (num = 0; isdigit(ch); ch = nextCh()) num = num * 10 + (ch - '0');
		return Token(IntNum, num);
	}
	else if (islower(ch)) {
		num = ch - 'a';
		ch = nextCh();
		return Token(VarName, num);
	}
	else {
		switch (ch) {
		case '(': kd = Lparen; break;
		case ')': kd = Rparen; break;
		case '+': kd = Plus; break;
		case '-': kd = Minus; break;
		case '*': kd = Multi; break;
		case '/': kd = Divi; break;
		case '=': kd = Assign; break;
		case '?': kd = Print; break;
		case '\0': kd = EofTkn; break;
		}
		ch = nextCh();
		return Token(kd);
	}
}

int nextCh() {
	if (*bufp == '\0') return '\0'; else return *bufp++;
}

void operate(TknKind op) {
	int d2 = pop(), d1 = pop();

	if (op == Divi && d2 == 0) { cout << " division by 0\n"; errF = 1; }
	if (errF) return;
	switch (op) {
	case Plus: push(d1 + d2); break;
	case Minus: push(d1 - d2); break;
	case Multi: push(d1*d2); break;
	case Divi: push(d1 / d2); break;
	}
}

void push(int n) {
	if (errF) return;
	if (stkct + 1 > STK_SIZ) { cout << "stack overflow\n"; exit(1); }
	stack[++stkct] = n;
}

int pop() {
	if (errF) return 1;
	if (stkct < 1) { cout << "stack underflow\n"; exit(1); }
	return stack[stkct--];
}

void chkTkn(TknKind kd) {
	if (token.kind != kd) errF = 1;
}