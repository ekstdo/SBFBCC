#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define U16MAX 65536
#define HEADER 0x627261696e66636b
#define INITIAL_BUFFER_SIZE 20

typedef enum {
	ADD_CONST = 0,
	SET_CONST = 1,
	MUL_CONST = 2,
	ADD_MUL = 3,
	ADD_CELL = 4,
	SUB_CELL = 5,
	LOAD = 6,
	LOAD_SWAP = 7,
	SHIFT = 8,
	SHIFT_BIG = 9,
	READ = 10,
	WRITE = 11,
	J = 12,
	JEZ = 13,
	JNEZ = 14,
	LABEL = 15,
	DEBUG_BREAK = 16,
	DEBUG_DATA = 17,
	MARK_EOF = 18
} Op;

typedef struct {
	signed int l;
	signed short int s;
	unsigned char b;
	unsigned char op;
} Opcode;

unsigned char tape[U16MAX];

int run_bytecode(Opcode* opcodes, int num_opcodes) {
	/* unsigned char* tape = mmap(NULL, U16MAX, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); */
	/* if (tape == MAP_FAILED) { */
	/* 	perror("failed to allocate memory"); */
	/* 	exit(1); */
	/* } */
	/* unsigned char* tape = calloc(U16MAX , sizeof(char)); */
	unsigned char* index = tape + U16MAX / 2 - 1;
	Opcode* pc = opcodes;
	unsigned char tmp;
	unsigned char reg = 0;


	static const void* table[] = {
		&&CASE_ADD_CONST,
		&&CASE_SET_CONST,
		&&CASE_MUL_CONST,
		&&CASE_ADD_MUL,
		&&CASE_ADD_CELL,
		&&CASE_SUB_CELL,
		&&CASE_LOAD,
		&&CASE_LOAD_SWAP,
		&&CASE_SHIFT,
		&&CASE_SHIFT_BIG,
		&&CASE_READ,
		&&CASE_WRITE,
		&&CASE_J,
		&&CASE_JEZ,
		&&CASE_JNEZ,
		&&CASE_LABEL,
		&&CASE_DEBUG_BREAK,
		&&CASE_DEBUG_DATA,
		&&CASE_MARK_EOF
	};
#define DISPATCH do { goto *table[(++pc)->op]; } while (0)
#define DISPATCH_NO_ADD do { goto *table[pc->op]; } while (0)
	pc -= 1;
	DISPATCH;
	CASE_LABEL:
	CASE_SHIFT_BIG:
	CASE_DEBUG_BREAK:
	CASE_DEBUG_DATA:
		DISPATCH;
	CASE_ADD_CONST:
		index[pc->l] += pc->b;
		DISPATCH;
	CASE_SET_CONST:
		index[pc->l] = pc->b;
		DISPATCH;
	CASE_MUL_CONST:
		index[pc->l] *= pc->b;
		DISPATCH;
	CASE_ADD_MUL:
		index[pc->l] += index[pc->l + pc->s] * pc->b;
		DISPATCH;
	CASE_ADD_CELL:
		index[pc->l] += index[pc->l + pc->s];
		DISPATCH;
	CASE_SUB_CELL:
		index[pc->l] -= index[pc->l + pc->s];
		DISPATCH;
	CASE_LOAD:
		reg = index[pc->l];
		DISPATCH;
	CASE_LOAD_SWAP:
		tmp = index[pc->l];
		index[pc->l] = reg;
		reg = tmp;
		DISPATCH;
	CASE_SHIFT:
		index += (signed int) pc->l;
		DISPATCH;
	CASE_READ:
		index[pc->l] = getchar();
		DISPATCH;
	CASE_WRITE:
		putchar(index[pc->l]);
		DISPATCH;
	CASE_JNEZ:
		if (*index) {
			pc = opcodes + pc->l;
			DISPATCH_NO_ADD;
		}
		DISPATCH;
	CASE_JEZ:
		if (*index == 0) {
			pc = opcodes + pc->l;
			DISPATCH_NO_ADD;
		}
		DISPATCH;
	CASE_J:
		pc = opcodes + pc->l;
		DISPATCH_NO_ADD;
	CASE_MARK_EOF:
		return 0;
}

void magic_number_check(FILE* fd) {
	unsigned long long int buffer;
	int result = fread(&buffer, sizeof(unsigned long long int), 1, fd);
	if (result != 1) {
		perror("missing header");
		exit(1);
	}
	if (buffer != HEADER) {
		perror("incorrect header");
		printf("actual header: %llx", buffer);
		exit(1);
	}
}

Opcode* read_opcodes(FILE* fd, size_t* num_opcodes) {
	Opcode* opcodes = malloc(sizeof(Opcode) * INITIAL_BUFFER_SIZE);
	size_t capacity = INITIAL_BUFFER_SIZE;
	size_t len = 0;
	while (!feof(fd)) {
		if (len == capacity) {
			opcodes = realloc(opcodes, sizeof(Opcode) * capacity * 2);
			capacity *= 2;
		}
		int result = fread(opcodes + len, sizeof(Opcode), 1, fd);
		if (result != 1) {
			printf("incomplete opcode: %x\n",  opcodes[len]);
		} else {
			len++;
		}
	}
	if (len == capacity) {
		opcodes = realloc(opcodes, sizeof(Opcode) * capacity * 2);
		capacity *= 2;
	}

	Opcode end = { .op = MARK_EOF };
	opcodes[len] = end;
	printf("read %zu opcodes in total\n", len);
	*num_opcodes = len;
	return opcodes;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		perror("missing filename");
		exit(1);
	}
	FILE* fd = fopen(argv[1], "rb");
	if (!fd) {
		perror("missing file");
		exit(1);
	}

	magic_number_check(fd);
	size_t num_opcodes = 0;
	Opcode* opcodes = read_opcodes(fd, &num_opcodes);
	run_bytecode(opcodes, num_opcodes);
	
	fclose(fd);
	return 0;
}
