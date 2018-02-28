#include <iostream>
#include <cstdlib>
#include <cctype>
using namespace std;

void polish(char *s);
int execute();
int getvalue(int ch);
int order(int ch);
void push(int n);
int pop();

#define STK_SIZ 20
int stack[STK_SIZ + 1];
int stkct;
char plsh_out[80];

int main() {
	char siki[80];
	int ans;

	cout << "�Է�: "; cin.getline(siki, 80);
	polish(siki);
	if (plsh_out[0] == '\0') exit(1);
	ans=execute();
	cout << "��ȯ: " << plsh_out << endl;
	cout << "���: " << ans << endl;

	return 0;
}

void polish(char *s) {
	int wk;
	char* out = plsh_out;

	stkct = 0;
	for (;;) {
		while (isspace(*s)) { ++s; }
		if(*s == '\0') {
			while (stkct > 0) {
				if ((*out++ = pop()) == '(') {
					puts("'('�� �ٸ��� ����\n"); exit(1);
				}
			}
			break;
		}
		if (islower(*s) || isdigit(*s)) {
			*out++ = *s++; continue;
		}
		switch (*s) {
		case '(':
			push(*s);
			break;
		case ')':
			while ((wk = pop()) != '(') {
				*out++ = wk;
				if (stkct == 0) { puts("'('�� �ٸ��� ����\n"); exit(1); }
			}
			break;
		default:
			if (order(*s) == -1) {
				cout << "�ٸ��� ���� ����: " << *s << endl; exit(1);
			}
			while (stkct > 0 && (order(stack[stkct]) >= order(*s))) {
				*out++ = pop();
			}
			push(*s);
		}
		s++;
	}
	*out = '\0';
}

int execute() {
	int d1, d2;
	char *s;

	stkct = 0;
	for (s = plsh_out; *s; s++) {
		if (islower(*s)) push(getvalue(*s));
		else if (isdigit(*s)) push(*s - '0');
		else {
			d2 = pop();
			d1 = pop();
			switch (*s) {
			case '+': push(d1 + d2); break;
			case '-': push(d1 - d2); break;
			case '*': push(d1*d2); break;
			case '/': if (d2 == 0) { puts("0���� ����"); exit(1); }
					  push(d1 / d2); break;
			}
		}
	}
	if (stkct != 1) { cout << "error\n"; exit(1); }
	return pop();
}

int getvalue(int ch) {
	if (islower(ch)) return ch - 'a' + 1;
	return 0;
}

int order(int ch) {
	switch (ch) {
	case'*': case'/': return 3;
	case'+': case '-': return 2;
	case '(': return 1;
	}
	return -1;
}

void push(int n) {
	if (stkct + 1 > STK_SIZ) { puts("stack overflow"); exit(1); }
	stack[++stkct] = n;
}

int pop() {
	if (stkct < 1) {
		puts("stack underflow"); exit(1);
	}
	return stack[stkct--];
}