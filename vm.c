#include <stdio.h>
#include <stdint.h>
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
	JNEZ_SHIFT = 15,
	LABEL = 16,
	DEBUG_BREAK = 17,
	DEBUG_DATA = 18,
	SKIP_LOOP = 19,
	MARK_EOF = 20
} Op;

char opnames[30][14] = {
	"ADD_CONST",
	"SET_CONST",
	"MUL_CONST",
	"ADD_MUL",
	"ADD_CELL",
	"SUB_CELL",
	"LOAD",
	"LOAD_SWAP",
	"SHIFT",
	"SHIFT_BIG",
	"READ",
	"WRITE",
	"J",
	"JEZ",
	"JNEZ",
	"JNEZ_SHIFT",
	"LABEL",
	"DEBUG_BREAK",
	"DEBUG_DATA",
	"SKIP_LOOP",
	"MARK_EOF"
};

typedef struct {
	int32_t l;
	int16_t s;
	uint8_t b;
	uint8_t op;
} ParsedOpcode;

typedef struct {
	uint8_t op;
	uint8_t b;
	int32_t v1;
	int32_t v2;
} BigOpcode;

int profile_jez_true = 0;
int profile_jez_false = 0;
int profile_jnez_true = 0;
int profile_jnez_false = 0;
/* uint8_t tape[U16MAX]; */
int profile_inst[U16MAX];
int profile_inst_type[30];
int profile_factor[256];

int run_bytecode(ParsedOpcode* opcodes, int num_opcodes) {
	uint8_t* tape = mmap(NULL, U16MAX, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (tape == MAP_FAILED) {
		perror("failed to allocate memory");
		exit(1);
	}
	/* uint8_t* tape = calloc(U16MAX , sizeof(char)); */
	uint8_t* index = tape + U16MAX / 2 - 1;
	ParsedOpcode* pc = opcodes;
	uint8_t tmp;
	uint8_t reg = 0;


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
		&&CASE_JNEZ_SHIFT,
		&&CASE_LABEL,
		&&CASE_DEBUG_BREAK,
		&&CASE_DEBUG_DATA,
		&&CASE_SKIP_LOOP,
		&&CASE_MARK_EOF
	};
#define DISPATCH do { profile_inst_type[pc->op] += 1; goto *table[pc->op]; } while (0)
	DISPATCH;

	CASE_ADD_CONST:
		index[pc->l] += pc->b;
		pc += 1;
		DISPATCH;
	CASE_SET_CONST:
		index[pc->l] = pc->b;
		pc += 1;
		DISPATCH;
	CASE_MUL_CONST:
		index[pc->l] *= pc->b;
		pc += 1;
		DISPATCH;
	CASE_ADD_MUL:
		index[pc->l] += pc->b * index[pc->s];
		profile_factor[pc->b] += 1;
		pc += 1;
		DISPATCH;
	CASE_ADD_CELL:
		index[pc->l] += index[pc->l + pc->s];
		pc += 1;
		DISPATCH;
	CASE_SUB_CELL:
		index[pc->l] -= index[pc->l + pc->s];
		pc += 1;
		DISPATCH;
	CASE_LOAD:
		reg = index[pc->l];
		pc += 1;
		DISPATCH;
	CASE_LOAD_SWAP:
		tmp = index[pc->l];
		index[pc->l] = reg;
		reg = tmp;
		pc += 1;
		DISPATCH;
	CASE_SHIFT:
		index += (signed int) pc->l;
		pc += 1;
		DISPATCH;
	CASE_READ:
		index[pc->l] = getchar();
		pc += 1;
		DISPATCH;
	CASE_WRITE:
		putchar(index[pc->l]);
		pc += 1;
		DISPATCH;;
	CASE_JNEZ_SHIFT:
		index += pc->s;
		// 92% success rate for this branch for mandelbrot.lbf
		if (__builtin_expect(!!*index, 1)) {
			pc = opcodes + pc->l;
		} else {
			pc += 1;
		}
		DISPATCH;
	CASE_JEZ:
		if (!*index) {
			pc = opcodes + pc->l;
		} else {
			pc += 1;
		}
		DISPATCH;

	CASE_SKIP_LOOP:
		while (*index) {
			index += pc->l;
		}
		pc += 1;
		DISPATCH;
	CASE_MARK_EOF:
		return 0;

	CASE_J:
		pc = opcodes + pc->l;
		DISPATCH;
	CASE_JNEZ:
		if (*index) {
			pc = opcodes + pc->l;
		} else {
			pc += 1;
		}
		DISPATCH;

	CASE_LABEL:
	CASE_SHIFT_BIG:
	CASE_DEBUG_BREAK:
	CASE_DEBUG_DATA:
		DISPATCH;
}

void magic_number_check(FILE* fd) {
	uint64_t buffer;
	int result = fread(&buffer, sizeof(unsigned long long int), 1, fd);
	if (result != 1) {
		perror("missing header");
		exit(1);
	}
	if (buffer != HEADER) {
		perror("incorrect header");
		printf("actual header: %lx", buffer);
		exit(1);
	}
}

ParsedOpcode* read_opcodes(FILE* fd, size_t* num_opcodes) {
	ParsedOpcode* opcodes = malloc(sizeof(ParsedOpcode) * INITIAL_BUFFER_SIZE);
	size_t capacity = INITIAL_BUFFER_SIZE;
	size_t len = 0;
	while (!feof(fd)) {
		if (len == capacity) {
			opcodes = realloc(opcodes, sizeof(ParsedOpcode) * capacity * 2);
			capacity *= 2;
		}
		int result = fread(opcodes + len, sizeof(ParsedOpcode), 1, fd);
		if (result != 1) {
			printf("incomplete opcode: %x\n",  opcodes[len]);
		} else {
			len++;
		}
	}
	if (len == capacity) {
		opcodes = realloc(opcodes, sizeof(ParsedOpcode) * capacity * 2);
		capacity *= 2;
	}

	ParsedOpcode end = { .op = MARK_EOF };
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
	ParsedOpcode* opcodes = read_opcodes(fd, &num_opcodes);
	fclose(fd);
	run_bytecode(opcodes, num_opcodes);
	
	printf("JNEZ Success: %d\nJNEZ Fail: %d\nJEZ Success: %d\nJEZ Fail: %d\n", profile_jnez_true, profile_jnez_false, profile_jez_true, profile_jez_false);
	for (int i = 0; i < MARK_EOF; i++) {
		printf("%s: %d\n", opnames[i], profile_inst_type[i]);
	}

	for (int i = 0; i < 256; i++) {
		if (profile_factor[i] != 0) {
			printf("%d:\t %d\n", i, profile_factor[i]);
		}
	}
	return 0;
}
