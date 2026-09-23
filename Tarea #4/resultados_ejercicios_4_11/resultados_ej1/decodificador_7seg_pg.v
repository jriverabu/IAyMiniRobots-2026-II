// Decodificador BCD -> 7 segmentos (cátodo común)
// Diseñado automáticamente por Programación Genética (DEAP), conjunto 'extendido'
module decodificador_7seg_pg (
    input  wire A, B, C, D,          // A = MSB
    output wire seg_a, seg_b, seg_c, seg_d, seg_e, seg_f, seg_g
);
    assign seg_a = (((D & ~A) ^ ~B) | C);
    assign seg_b = ~(B & (C ^ D));
    assign seg_c = ((B | D) | ~C);
    assign seg_d = ((D & A) ^ ((C & ~B) | ((D ^ ~B) ^ C)));
    assign seg_e = ~((B & ~C) | D);
    assign seg_f = ~((~B & (C | (~A & D))) | (C & D));
    assign seg_g = (((C & D) ^ (C ^ A)) | (B ^ C));
endmodule
