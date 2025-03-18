#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define U16MAX 65536
#define INITIAL_INDEX (U16MAX / 2 - 1)
#define HEADER 0x627261696e66636b
#define INITIAL_BUFFER_SIZE 20

/* #define PROFILE_ALL */
/* #define ENABLE_DEBUGGER */



typedef enum {
	ADD_CONST = 0,
	SET_CONST = 1,
	MUL_CONST = 2,
	ADD_MUL = 3,
	ADD_CELL = 4,
	ADDN_SET0 = 5,
	SUB_CELL = 6,
	LOAD = 7,
	LOAD_SWAP = 8,
	SWAP = 9,
	LOAD_MUL = 10,
	ADD_STORE_MUL = 11,
	SQADD_REG = 12,
	SHIFT = 13,
	SHIFT_BIG = 14,
	READ = 15,
	WRITE = 16,
	J = 17,
	JEZ = 18,
	JNEZ = 19,
	JEZ_SHIFT = 20,
	JNEZ_SHIFT = 21,
	SKIP_LOOP = 22,
	MARK_EOF = 23,
	DEBUG_BREAK = 24,
	LABEL = 128,
	DEBUG_DATA = 129,
} Op;

char opnames[256][14] = {
	"ADD_CONST",
	"SET_CONST",
	"MUL_CONST",
	"ADD_MUL",
	"ADD_CELL",
	"ADDN_SET0",
	"SUB_CELL",
	"LOAD",
	"LOAD_SWAP",
	"SWAP",
	"LOAD_MUL",
	"ADD_STORE_MUL",
	"SQADD_REG",
	"SHIFT",
	"SHIFT_BIG",
	"READ",
	"WRITE",
	"J",
	"JEZ",
	"JNEZ",
	"JEZ_SHIFT",
	"JNEZ_SHIFT",
	"SKIP_LOOP",
	"MARK_EOF",
	"DEBUG_BREAK",
	"", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
	"LABEL",
	"DEBUG_DATA",
};

char* rtrim(char* s) {
	size_t length = strlen(s);
	char* back = s + length;
	while (isspace(*--back)) {}
	*(back + 1) = '\0';
	return s;
}

char* ltrim(char *s) {
	while (isspace(*s)) {s += 1;}
	return s;
}

char* trim(char* s) {
	if (s == NULL) {
		fprintf(stderr, "%s", "Trimming met with a NULL string\n");
		return s;
	}
	return rtrim(ltrim(s));
}

typedef struct {
	int32_t l;
	int16_t s;
	uint8_t b;
	uint8_t op;
} ParsedOpcode;


char* print_instruction(ParsedOpcode* op);

unsigned num_bits(int x) {
	unsigned bits, var = (x < 0) ? -x : x;
	for(bits = 0; var != 0; ++bits) var >>= 1;
	return bits;
}

unsigned char colors[20][3] = {
    {0xf4, 0x86, 0xde},
    {0xff, 0x84, 0xd4},
    {0xff, 0x85, 0xc9},
    {0xff, 0x8d, 0xbd},
    {0xff, 0x95, 0xb4},
    {0xff, 0x9b, 0xad},
    {0xff, 0xa1, 0xa6},
    {0xff, 0xa7, 0x9f},
    {0xff, 0xac, 0x98},
    {0xff, 0xb1, 0x90},
    {0xff, 0xb6, 0x86},
    {0xff, 0xbb, 0x77},
    {0xff, 0xc0, 0x61},
    {0xff, 0xc8, 0x53},
    {0xf9, 0xd1, 0x4f},
    {0xed, 0xda, 0x4f},
    {0xdf, 0xe3, 0x52},
    {0xd0, 0xeb, 0x59},
    {0xbf, 0xf3, 0x63},
    {0xac, 0xfa, 0x70},
};

#define MIN(x, y) (x > y ? y : x)
#define MAX(x, y) (x > y ? x : y)

char* get_grad_factor(int x, float factor){
	unsigned char* col_this = colors[19-MIN((int) (num_bits(x) * factor), 20 - 1)];
	char* buffer = malloc(20);
	sprintf(buffer, "\x1b[38;2;%d;%d;%dm", col_this[0], col_this[1], col_this[2]);
	return buffer;
}

char* get_grad(int x){
	unsigned char* col_this = colors[19-MIN(num_bits(x), 20 - 1)];
	char* buffer = malloc(20);
	sprintf(buffer, "\x1b[38;2;%d;%d;%dm", col_this[0], col_this[1], col_this[2]);
	return buffer;
}

#ifdef PROFILE_ALL

int profile_jez_true = 0;
int profile_jez_false = 0;
int profile_jnez_true = 0;
int profile_jnez_false = 0;
int profile_jezs_true = 0;
int profile_jezs_false = 0;
int profile_jnezs_true = 0;
int profile_jnezs_false = 0;
/* uint8_t tape[U16MAX]; */
int profile_inst[U16MAX];
int profile_inst_type[MARK_EOF];
int profile_add_mul[256];
int profile_add_const[256];
int profile_set_const[256];

int profile_from_to[MARK_EOF][MARK_EOF];

#endif /* ifdef PROFILE_ALL */



typedef struct {
	uint16_t start_col;
	uint16_t start_line;
	int8_t offset_col;
	uint8_t offset_line;
	uint16_t num;
} ParseDebugPos;

typedef struct {
	ParsedOpcode* opcodes;
	int num_opcodes;
	ParseDebugPos* debug_pos; // list of Debug pos
	int num_debug_pos;
	ParseDebugPos** translated; // repeated pointers to Debug positions
	char* src; // src code
	char** lines; // src lines
	size_t* line_lengths;
	int num_lines;
} ParseOutput;



int run_bytecode(ParseOutput p) {
	ParsedOpcode* opcodes = p.opcodes;
	int num_opcodes = p.num_opcodes;
	uint8_t* tape = mmap(NULL, U16MAX, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (tape == MAP_FAILED) {
		perror("failed to allocate memory");
		exit(1);
	}
	/* uint8_t* tape = calloc(U16MAX , sizeof(char)); */
	register uint8_t* index = tape + INITIAL_INDEX;
	register ParsedOpcode* pc = opcodes;
	uint8_t tmp;
	register uint8_t reg = 0;

#ifdef ENABLE_DEBUGGER
	int debug_counter = -1;
	ParsedOpcode* last_pc = opcodes;
	#define PC(action) do { last_pc = pc; pc action; debug_counter -= 1; if (!debug_counter) goto DEBUG; } while (0);
#else
	#define PC(action) do { pc action; } while(0);
#endif /* ifdef ENABLE_DEBUGGER */


#ifdef PROFILE_ALL
#define PROFILE_INST do { profile_inst_type[pc->op] += 1; profile_inst[pc - opcodes] += 1; profile_from_to[(pc - 1)->op][pc->op] += 1; } while (0);
#define PROFILE_INC(x) do { x += 1; } while (0);
#else
#define PROFILE_INST
#define PROFILE_INC(x) 
#endif


	static const void* table[] = {
		&&CASE_ADD_CONST,
		&&CASE_SET_CONST,
		&&CASE_MUL_CONST,
		&&CASE_ADD_MUL,
		&&CASE_ADD_CELL,
		&&CASE_ADDN_SET0,
		&&CASE_SUB_CELL,
		&&CASE_LOAD,
		&&CASE_LOAD_SWAP,
		&&CASE_SWAP,
		&&CASE_LOAD_MUL,
		&&CASE_ADD_STORE_MUL,
		&&CASE_SQADD_REG,
		&&CASE_SHIFT,
		&&CASE_SHIFT_BIG,
		&&CASE_READ,
		&&CASE_WRITE,
		&&CASE_J,
		&&CASE_JEZ,
		&&CASE_JNEZ,
		&&CASE_JEZ_SHIFT,
		&&CASE_JNEZ_SHIFT,
		&&CASE_SKIP_LOOP,
		&&CASE_MARK_EOF,
		&&CASE_DEBUG_BREAK,
	};


#define DISPATCH do { \
	PROFILE_INST \
	goto *table[pc->op]; \
} while (0)

	DISPATCH;

	CASE_ADD_CONST: PROFILE_INC(profile_add_const[pc->b]);
		index[pc->l] += pc->b;
		PC(+= 1);
		DISPATCH;
	CASE_SET_CONST: PROFILE_INC(profile_set_const[pc->b]);
		index[pc->l] = pc->b;
		PC(+= 1);
		DISPATCH;
	CASE_MUL_CONST:
		index[pc->l] *= pc->b;
		PC(+= 1);
		DISPATCH;
	CASE_ADD_MUL: PROFILE_INC(profile_add_mul[pc->b]);
		index[pc->l] += pc->b * index[pc-> l + pc->s];
		PC(+= 1);
		DISPATCH;
	CASE_ADD_CELL:
		index[pc->l] += index[pc->l + pc->s];
		PC(+= 1);
		DISPATCH;
	CASE_ADDN_SET0:
		index[pc->l] += index[pc->l + pc->s];
		index[pc->l + pc->s] = 0;
		PC(+= 1);
		DISPATCH;
	CASE_SUB_CELL:
		index[pc->l] -= index[pc->l + pc->s];
		PC(+= 1);
		DISPATCH;
	CASE_LOAD:
		reg = index[pc->l];
		PC(+= 1);
	CASE_LOAD_SWAP:
		do {
			tmp = index[pc->l];
			index[pc->l] = reg;
			reg = tmp;
			PC(+= 1);
		} while (pc->op == LOAD_SWAP);
		DISPATCH;
	CASE_SWAP:
		tmp = index[pc->l + pc->s];
		index[pc->l + pc->s] = index[pc->l];
		index[pc->l] = tmp;
		PC(+= 1);
		DISPATCH;
	CASE_SHIFT:
		index += (signed int) pc->l;
		PC(+= 1);
		DISPATCH;
	CASE_READ:
		index[pc->l] = getchar();
		PC(+= 1);
		DISPATCH;
	CASE_WRITE:
		putchar(index[pc->l]);
		PC(+= 1);
		DISPATCH;
	CASE_JEZ_SHIFT:
		index += pc->s;
		if (!*index) {
			PC(= opcodes + pc->l);
			PROFILE_INC(profile_jezs_true);
			DISPATCH;
		} else {
			PC(+= 1);
			PROFILE_INC(profile_jezs_false);
			DISPATCH;
		}
	CASE_JNEZ_SHIFT:
		index += pc->s;
		// 92% success rate for this branch for mandelbrot.lbf
		if (__builtin_expect(!!*index, 1)) {
			PC(= opcodes + pc->l);
			PROFILE_INC(profile_jnezs_true);
			DISPATCH;
		} else {
			PC(+= 1);
			PROFILE_INC(profile_jnezs_false);
			DISPATCH;
		}
	CASE_JEZ:
		if (!*index) {
			PC(= opcodes + pc->l);
			PROFILE_INC(profile_jez_true);
			DISPATCH;
		} else {
			PC(+= 1);
			PROFILE_INC(profile_jez_false);
			DISPATCH;
		}

	CASE_SKIP_LOOP:
		while (*index) {
			index += pc->l;
		}
		PC(+= 1);
		DISPATCH;
	CASE_MARK_EOF:
		return 0;
	CASE_JNEZ:
		// around 66% success rate, so not enough for __builtin_expect
		if (*index) {
			PROFILE_INC(profile_jnez_true);
			CASE_J:
			PC(= opcodes + pc->l);
			DISPATCH;
		} else {
			PROFILE_INC(profile_jnez_false);
			PC(+= 1);
			DISPATCH;
		}
	CASE_DEBUG_BREAK:
#ifdef ENABLE_DEBUGGER
		goto DEBUG;
#endif
	CASE_SHIFT_BIG:
		PC(+= 1);
		DISPATCH;

	CASE_LOAD_MUL:
		reg = pc->b * index[pc->l];
		PC(+= 1);
		// is always followed by ADD_STORE_MUL
	CASE_ADD_STORE_MUL:
		index[pc->l] = reg * pc->b;
		PC(+= 1);
		DISPATCH;
	CASE_SQADD_REG:
		reg += 1;
		tmp = index[pc->l];
		reg = tmp & 1 ? (reg >> 1) * tmp : (tmp >> 1) * reg;
		PC(+= 1);
		// is always followed by ADD_STORE_MUL
		index[pc->l] = reg * pc->b;
		PC(+= 1);
		DISPATCH;
#ifdef ENABLE_DEBUGGER
	DEBUG:
		printf("\x1b[32m[ENTERED DEBUG MODE]\n");
		printf("\treg:    %d\n", reg);
		printf("\tindex:  %ld\n", index - tape - INITIAL_INDEX);
		printf("\ttape [%ld to %ld]: [", index - tape - 10 - INITIAL_INDEX, index - tape + 10 - INITIAL_INDEX);
		for (int i = 10; i > 0; i --) 
			printf("%d, ", *(index - i));
		printf("< %d >, ", *index);
		for (int i = 1; i < 10; i ++) 
			printf("%d, ", *(index + i));
		printf("%d", *(index + 10));
		printf("]\n");

		for (int i = -5; i <= 5; i++) {
			if (last_pc - opcodes + i < 0) { continue; }
			if (last_pc - opcodes + i >= num_opcodes) { continue; }
			char* co = print_instruction(last_pc + i);
			if (i == 0) {
				printf("\x1b[0m\x1b[42m");
			}
			printf("%5ld: %s", last_pc - opcodes + i, co);
			if (i == 0) {
				printf("\x1b[0m\x1b[32m");
			}
			printf("\n");
			free(co);
		}

		printf("\n\n");

		ParseDebugPos* parsed_opcode = p.translated[last_pc - opcodes];
		printf("RANGE: %d-%d, %d-%d\n", parsed_opcode->start_line, parsed_opcode->start_line + parsed_opcode->offset_line, parsed_opcode->start_col, parsed_opcode->start_col + parsed_opcode->offset_col);
		size_t start_line_length = p.line_lengths[parsed_opcode->start_line];
		size_t end_line = parsed_opcode->start_line + parsed_opcode->offset_line;
		size_t end_line_length = p.line_lengths[parsed_opcode->start_line + parsed_opcode->offset_line];
		size_t end_col = parsed_opcode->start_col + parsed_opcode->offset_col;
		char* buffer = malloc(sizeof(char) * start_line_length);
		strncpy(buffer, p.lines[parsed_opcode->start_line], parsed_opcode->start_col);
		buffer[parsed_opcode->start_col] = '\0';
		printf("%s\x1b[0m\x1b[42m", buffer);


		if (parsed_opcode->offset_line == 0) {
			strncpy(buffer, p.lines[parsed_opcode->start_line] + parsed_opcode->start_col, parsed_opcode->offset_col + 1);
			buffer[parsed_opcode->offset_col+1] = '\0';
			printf("%s", buffer);
			printf("\x1b[0m\x1b[32m");
			strcpy(buffer, p.lines[parsed_opcode->start_line] + end_col + 1);
			printf("%s\n", buffer);
		} else {
			strcpy(buffer, p.lines[parsed_opcode->start_line] + parsed_opcode->start_col);
			printf("%s\n", buffer);
			for (uint16_t line = parsed_opcode->start_line + 1; line < parsed_opcode->start_line + parsed_opcode->offset_line; line++) {
				printf("%s\n", p.lines[line]);
			}
			buffer = realloc(buffer, sizeof(char) * p.line_lengths[end_line]);
			strncpy(buffer, p.lines[end_line], end_col);
			buffer[end_col+1] = '\0';
			printf("%s\x1b[0m\x1b[32m", buffer);
			strcpy(buffer, p.lines[end_line] + end_col + 1);
			printf("%s\n", buffer);
		}

		do {
			char buffer[50];
		DEBUG_START:
			printf("\x1b[32mDEBUG[%ld @ %s] > ", pc - opcodes, opnames[last_pc->op]);
			fflush(stdout);
			fgets(buffer, 49, stdin);

			int num_args = 0;
			char debug_op;
			int arg1;
			int arg2;
			
			char* parse_err;

			for (char* token = strtok(buffer, " "); token != NULL; token = strtok(NULL, " ")) {
				if (num_args == 0) {
					debug_op = token[0];
				} else if (num_args == 1) {
					arg1 = strtol(token, &parse_err, 10);
					if (*parse_err != 0 && !isspace(*parse_err)) { printf("couldn't parse first number!\n"); goto DEBUG_START; }
				} else if (num_args == 2) {
					arg2 = strtol(token, &parse_err, 10);
					if (*parse_err != 0 && !isspace(*parse_err)) { printf("couldn't parse second number!\n"); goto DEBUG_START; }
				} else {
					num_args ++;
					break;
				}
				
				num_args ++;
			}

			switch (debug_op) {
				case 'p':
					goto DEBUG;
				case 'r':
					if (num_args == 2) {

					} else {
						printf("2 or 3 arguments needed!\n");
					}
					break;
				case 'w':
					if (num_args != 3) {
						printf("2 arguments needed!\n");
						break;
					}
					tape[INITIAL_INDEX + arg1] = arg2;
					printf("WRITTEN %d [%c] TO t[%d]\n", arg2, arg2, arg1);
					break;
				case 's':
				case '\n':
					debug_counter = num_args == 2 ? arg1 : 1;
					printf("STEP %d\x1b[0m\n", debug_counter);
					goto QUIT_DEBUG;
				case 'q':
					debug_counter = -1;
					printf("EXIT DEBUG\x1b[0m\n");
					goto QUIT_DEBUG;
				default:
					printf("Unknown debug op\n Use p to print an overview\n     r <range>     to read parts of the tape\n     w <to> <val>  to write a value to the tape\n     s <n>         to step n steps\n");
					break;
			}
		} while (1);
	QUIT_DEBUG:
		if (pc->op == DEBUG_BREAK)
			pc += 1;
		DISPATCH;
#endif /* ifdef ENABLE_DEBUGGER */
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



int is_end_pos(ParseDebugPos* p) {
	return p->start_col == 0xFFFF && p->start_line == 0xFFFF && (uint8_t) p->offset_col == 0xFF && p->offset_line == 0xFF && p->num == 0xFFFF;
}

#define INITIALIZE_BUFFER(x, type) type* x = malloc(sizeof(type) * INITIAL_BUFFER_SIZE); size_t x ## _capacity = INITIAL_BUFFER_SIZE; size_t num_ ## x = 0;
#define DOUBLE_BUFFER(x, type) if (num_ ## x == x ## _capacity ) { x ## _capacity *= 2; x = realloc(x, sizeof(type) * x ## _capacity); }

ParseOutput read_opcodes(FILE* fd) {

	INITIALIZE_BUFFER(opcodes, ParsedOpcode);
	while (!feof(fd)) {
		DOUBLE_BUFFER(opcodes, ParsedOpcode);
		int result = fread(opcodes + num_opcodes, sizeof(ParsedOpcode), 1, fd);

		if (opcodes[num_opcodes].op == MARK_EOF) {
			break;
		}
		if (result != 1) {
			printf("incomplete opcode: %x\n",  opcodes[num_opcodes]);
		} else {
			num_opcodes++;
		}

	}

	INITIALIZE_BUFFER(debug_pos, ParseDebugPos);
	num_debug_pos = 0;
	INITIALIZE_BUFFER(translated, ParseDebugPos*);
	num_translated = 0;
	while (!feof(fd)) {
		DOUBLE_BUFFER(debug_pos, ParseDebugPos);

		int result = fread(debug_pos + num_debug_pos, sizeof(ParseDebugPos), 1, fd);
		if (is_end_pos(debug_pos + num_debug_pos)) {
			break;
		}

		if (result != 1) {
			printf("incomplete debugcode: %x\n", debug_pos[num_debug_pos]);
		} else {
			num_debug_pos++;
		}
	}

	for (size_t e = 0; e < num_debug_pos; e++){
		for (int i = 0; i < debug_pos[e].num; i++) {
			DOUBLE_BUFFER(translated, ParseDebugPos*);

			translated[num_translated] = debug_pos + e;
			num_translated++;
		}
	}


	if (num_translated != num_opcodes) {
		printf("Num translated: %ld, num_opcodes: %ld\n", num_translated, num_opcodes);
		perror("Missing debug data!");
		exit(1);
	}

	INITIALIZE_BUFFER(src, char);
	while (!feof(fd)) {
		DOUBLE_BUFFER(src, char);
		fread(src + num_src, sizeof(char), 1, fd);
		num_src++;
	}

	INITIALIZE_BUFFER(lines, char*);
	INITIALIZE_BUFFER(line_lengths, size_t);
	// inefficient, as it's being iterated 3 times
	for (char* line = strtok(src, "\n"); line != NULL; line = strtok(NULL, "\n")) {
		DOUBLE_BUFFER(lines, char*);
		lines[num_lines++] = line;
		DOUBLE_BUFFER(line_lengths, size_t);
		line_lengths[num_line_lengths++] = strlen(line);
	}




	printf("read %zu opcodes in total\n", num_opcodes);
	ParseOutput out = {
		.opcodes = opcodes,
		.num_opcodes = num_opcodes,
		.num_debug_pos = num_debug_pos,
		.num_lines = num_lines,
		.translated = translated,
		.src = src,
		.lines = lines,
		.debug_pos = debug_pos,
		.line_lengths = line_lengths
	};
	return out;
}

char* print_instruction(ParsedOpcode* op) {
	char* buffer = malloc(64);
	switch (op->op) {
		case ADD_CONST:
			sprintf(buffer, "ADD_CONST       /     t[%d] += %d;", op->l, op->b);
			break;
		case SET_CONST:
			sprintf(buffer, "SET_CONST       /     t[%d]  = %d;", op->l, op->b);
			break;
		case MUL_CONST:
			sprintf(buffer, "MUL_CONST       /     t[%d] *= %d;", op->l, op->b);
			break;
		case LABEL:
			sprintf(buffer, "LABEL           / label %d;", op->l);
			break;
		case J:
			sprintf(buffer, "J               /     j %d;", op->l);
			break;
		case JEZ:
			sprintf(buffer, "JEZ             /     jez %d;", op->l);
			break;
		case JNEZ:
			sprintf(buffer, "JNEZ            /     jnez %d;", op->l);
			break;
		case JEZ_SHIFT:
			sprintf(buffer, "JEZ_SHIFT      /     t += %d; jnez %d;", op->s, op->l);
			break;
		case JNEZ_SHIFT:
			sprintf(buffer, "JNEZ_SHIFT      /     t += %d; jnez %d;", op->s, op->l);
			break;
		case WRITE:
			sprintf(buffer, "WRITE           /     putchar(t[%d]);", op->l);
			break;
		case READ:
			sprintf(buffer, "READ            /     t[%d] = getchar();", op->l);
			break;
		case SHIFT:
			sprintf(buffer, "SHIFT           /     t += %d;", op->l);
			break;
		case SHIFT_BIG:
			sprintf(buffer, "SHIFT_BIG       /     t += %d * 32768;", op->l);
			break;
		case LOAD:
			sprintf(buffer, "LOAD            /     r = t[%d];", op->l);
			break;
		case LOAD_SWAP:
			sprintf(buffer, "LOAD_SWAP       /     r, t[%d] = t[%d], r;", op->l, op->l);
			break;
		case SWAP:
			sprintf(buffer, "SWAP            /     t[%d], t[%d] = t[%d], t[%d];", op->l, op->l + op->s, op->l + op->s, op->l);
			break;
		case ADD_CELL:
			sprintf(buffer, "ADD_CELL        /     t[%d] += t[%d];", op->l, op->l + op->s);
			break;
		case ADDN_SET0:
			sprintf(buffer, "ADDN_SET0       /     t[%d] += t[%d]; t[%d] = 0;", op->l, op->l + op->s, op->l + op->s);
			break;
		case SUB_CELL:
			sprintf(buffer, "SUB_CELL        /     t[%d] -= t[%d];", op->l, op->l + op->s);
			break;
		case ADD_MUL:
			sprintf(buffer, "ADD_MUL         /     t[%d] += %d * t[%d];", op->l, op->b, op->l + op->s);
			break;
		case DEBUG_BREAK:
			sprintf(buffer, "DEBUG_BREAK  !!!");
			break;
		case SKIP_LOOP:
			sprintf(buffer, "SKIP_LOOP       /     while (t[0]) { t += %d }", op->l);
			break;
	}
	return buffer;
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
	ParseOutput parsed = read_opcodes(fd);
	fclose(fd);

	run_bytecode(parsed);
	free(parsed.translated);
	free(parsed.debug_pos);
	free(parsed.opcodes);
	free(parsed.src);
	free(parsed.lines);
	free(parsed.line_lengths);
	
#ifdef PROFILE_ALL
	for (int i = 0; i < parsed.num_opcodes; i++) {
		char* inst = print_instruction(&parsed.opcodes[i]);
		char* col = get_grad(profile_inst[i] >> 2);
		printf("%5d: %-64s  -> #%s%d\x1b[0m\n", i, inst, col, profile_inst[i]);
		free(inst);
		free(col);
	}
	for (int i = 0; i < MARK_EOF; i++) {
		char* col = get_grad_factor(profile_inst_type[i] >> 20, 2);
		printf("%-15s: %s%d\x1b[0m\n", opnames[i], col, profile_inst_type[i]);
		free(col);
		if (i == ADD_MUL) {
			for (int i = 0; i < 256; i++) {
				if (profile_add_mul[i] != 0) {
					char* col = get_grad(profile_inst_type[i] >> 4);
					printf("\t%5d:\t %s%d\x1b[0m\n", i, col, profile_add_mul[i]);
					free(col);
				}
			}
		} else if (i == SET_CONST) {
			for (int i = 0; i < 256; i++) {
				if (profile_set_const[i] != 0) {
					char* col = get_grad(profile_inst_type[i] >> 4);
					printf("\t%5d:\t %s%d\x1b[0m\n", i, col, profile_set_const[i]);
					free(col);
				}
			}
		} else if (i == ADD_CONST) {
			for (int i = 0; i < 256; i++) {
				if (profile_add_const[i] != 0) {
					char* col = get_grad(profile_inst_type[i] >> 4);
					printf("\t%5d:\t %s%d\x1b[0m\n", i, col, profile_add_const[i]);
					free(col);
				}
			}
		} else if (i == JNEZ) {
			printf("\tJNEZ Success: %d\n\tJNEZ Fail: %d\n", profile_jnez_true, profile_jnez_false);
		} else if (i == JEZ) {
			printf("\tJEZ Success: %d\n\tJEZ Fail: %d\n", profile_jez_true, profile_jez_false);
		} else if (i == JNEZ_SHIFT) {
			printf("\tJNEZ SHIFT Success: %d\n\tJNEZ SHIFT Fail: %d\n", profile_jnezs_true, profile_jnezs_false);
		} else if (i == JEZ_SHIFT) {
			printf("\tJNEZ SHIFT Success: %d\n\tJNEZ SHIFT Fail: %d\n", profile_jezs_true, profile_jezs_false);
		}
	}

	printf("from / to   ");
	for (int i = 0; i <= SKIP_LOOP; i++) {
		if (profile_inst_type[i] == 0) continue;
		printf("%10s", opnames[i]);
	}
	printf("\n");
	for (int i = 0; i <= SKIP_LOOP; i++) {
		if (profile_inst_type[i] == 0) continue;
		printf("%-8s:", opnames[i]);
		for (int j = 0; j <= SKIP_LOOP; j++) {
			if (profile_inst_type[j] == 0) continue;
			int val = profile_from_to[i][j];
			char* col = get_grad_factor(val>>10, 1.1);
			printf("%s%10d\x1b[0m", col, val);
			free(col);
		}
		printf("\n");
	}
	printf("\n");
#endif /* ifdef PROFILE_ALL */


	return 0;
}
