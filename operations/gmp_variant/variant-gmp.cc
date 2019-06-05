#include <gmp.h>
#include <string>
#include <iostream>

int main() {
    mpz_t int1;
    mpz_t int2;
    mpz_t res;

    mpz_init_set_ui(int1, 50000);
    mpz_init_set_ui(int2, 60000);
    mpz_init(res);

    mpz_add(res, int1, int2);

    gmp_printf("Num to be exported: %Zd\n", res);

    size_t count_p;

    char * p = (char *)mpz_export(NULL, &count_p, 1, sizeof(unsigned long), 0, 0, res);

    int total_size = count_p * sizeof(unsigned long);

    std::string s(p, total_size);

    mpz_t imp;

    mpz_init(imp);

    mpz_import(imp, count_p, 1, sizeof(unsigned long), 0, 0, s.c_str());

    gmp_printf("Imported Num: %Zd\n", imp);
}