using StructuralIdentifiability
using CSV

MA1 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)/(V(t)+k_T)-0.011*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MA1_identifiability = assess_identifiability(MA1)
println(MA1_identifiability)

MA2 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)/(V(t)+k_T)-c_T*(1/(1+V(t)^2))-0.011*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MA2_identifiability = assess_identifiability(MA2)
println(MA2_identifiability)


MA3 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)/(V(t)+k_T)-d_T*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MA3_identifiability = assess_identifiability(MA3)
println(MA3_identifiability)


MA4 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)/(V(t)+k_T)-c_T*(1/(1+V(t)^2))-d_T*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MA4_identifiability = assess_identifiability(MA4)
println(MA4_identifiability)


MB1 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - 0.011*T(t) + r*T(t)*(V(t)/(V(t)+k_T)),
    y1(t) = V(t),
    y2(t) = T(t)
)


MB1_identifiability = assess_identifiability(MB1)
println(MB1_identifiability)

MB2 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - 0.011*T(t) + r*T(t)*(V(t)/(V(t)+k_T)) - c_T*(1/(1+V(t)^2)),
    y1(t) = V(t),
    y2(t) = T(t)
)

MB2_identifiability = assess_identifiability(MB2)
println(MB2_identifiability)

MB3 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - d_T*T(t) + r*T(t)*(V(t)/(V(t)+k_T)),
    y1(t) = V(t),
    y2(t) = T(t)
)


MB3_identifiability = assess_identifiability(MB3)
println(MB3_identifiability)

MB4 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - d_T*T(t) + r*T(t)*(V(t)/(V(t)+k_T)) - c_T*(1/(1+V(t)^2)),
    y1(t) = V(t),
    y2(t) = T(t)
)

MB4_identifiability = assess_identifiability(MB4)
println(MB4_identifiability)

MC1 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)-0.011*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MC1_identifiability = assess_identifiability(MC1)
println(MC1_identifiability)

MC2 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)-c_T*(1/(1+V(t)^2))-0.011*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)


MC2_identifiability = assess_identifiability(MC2)
println(MC2_identifiability)

MC3 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)-d_T*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MC3_identifiability = assess_identifiability(MC3)
println(MC3_identifiability)

MC4 = @ODEmodel(
    V'(t) = p*V(t)*(1-V(t)/k_V) - c_V*V(t)*T(t),
    T'(t) = s_T + r*T(t)*V(t)-c_T*(1/(1+V(t)^2))-d_T*T(t),
    y1(t) = V(t),
    y2(t) = T(t)
)


MC4_identifiability = assess_identifiability(MC4)
println(MC4_identifiability)

MD1 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - 0.011*T(t) + r*T(t)*V(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MD1_identifiability = assess_identifiability(MD1)
println(MD1_identifiability)

MD2 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - 0.011*T(t) + r*T(t)*V(t) - c_T*(1/(1+V(t)^2)),
    y1(t) = V(t),
    y2(t) = T(t)
)

MD2_identifiability = assess_identifiability(MD2)
println(MD2_identifiability)

MD3 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - d_T*T(t) + r*T(t)*V(t),
    y1(t) = V(t),
    y2(t) = T(t)
)

MD3_identifiability = assess_identifiability(MD3)
println(MD3_identifiability)

MD4 = @ODEmodel(
    U'(t) = -beta*U(t)*V(t),
    I'(t) = beta*U(t)*V(t)-d_I*T(t)*I(t),
    V'(t) = p*I(t)-c*V(t),
    T'(t) = s_T - d_T*T(t) + r*T(t)*V(t) - c_T*(1/(1+V(t)^2)),
    y1(t) = V(t),
    y2(t) = T(t)
)

MD4_identifiability = assess_identifiability(MD4)
println(MD4_identifiability)

DESTINATION_FOLDER = "StructuralIdentifiabilityResults/"


println("MA1")
println(MA1_identifiability)
MA1_dict = Dict()
for i in keys(MA1_identifiability)
    MA1_dict[i]=MA1_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MA1_Identifiability.csv"),MA1_dict)

println()


println()
println("MA2")
println(MA2_identifiability)
MA2_dict = Dict()
for i in keys(MA2_identifiability)
    MA2_dict[i]=MA2_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MA2_Identifiability.csv"),MA2_dict)

println()

println("MA3")
println(MA3_identifiability)
MA3_dict = Dict()
for i in keys(MA3_identifiability)
    MA3_dict[i]=MA3_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MA3_Identifiability.csv"),MA3_dict)

println()

println("MA4")
println(MA4_identifiability)
MA4_dict = Dict()
for i in keys(MA4_identifiability)
    MA4_dict[i]=MA4_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MA4_Identifiability.csv"),MA4_dict)

println()

println("MB1")
println(MB1_identifiability)
MB1_dict = Dict()
for i in keys(MB1_identifiability)
    MB1_dict[i]=MB1_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MB1_Identifiability.csv"),MB1_dict)

println()

println("MB2")
println(MB2_identifiability)
MB2_dict = Dict()
for i in keys(MB2_identifiability)
    MB2_dict[i]=MB2_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MB2_Identifiability.csv"),MB2_dict)

println()

println("MB3")
println(MB3_identifiability)
MB3_dict = Dict()
for i in keys(MB3_identifiability)
    MB3_dict[i]=MB3_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MB3_Identifiability.csv"),MB3_dict)

println()

println("MB4")
println(MB4_identifiability)
MB4_dict = Dict()
for i in keys(MB4_identifiability)
    MB4_dict[i]=MB4_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MB4_Identifiability.csv"),MB4_dict)

println()

println("MC1")
println(MC1_identifiability)
MC1_dict = Dict()
for i in keys(MC1_identifiability)
    MC1_dict[i]=MC1_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MC1_Identifiability.csv"),MC1_dict)

println()

println("MC2")
println(MC2_identifiability)
MC2_dict = Dict()
for i in keys(MC2_identifiability)
    MC2_dict[i]=MC2_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MC2_Identifiability.csv"),MC2_dict)

println()

println("MC3")
println(MC3_identifiability)
MC3_dict = Dict()
for i in keys(MC3_identifiability)
    MC3_dict[i]=MC3_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MC3_Identifiability.csv"),MC3_dict)

println()

println("MC4")
println(MC4_identifiability)
MC4_dict = Dict()
for i in keys(MC4_identifiability)
    MC4_dict[i]=MC4_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MC4_Identifiability.csv"),MC4_dict)

println()

println("MD1")
println(MD1_identifiability)
MD1_dict = Dict()
for i in keys(MD1_identifiability)
    MD1_dict[i]=MD1_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MD1_Identifiability.csv"),MD1_dict)

println()

println("MD2")
println(MD2_identifiability)
MD2_dict = Dict()
for i in keys(MD2_identifiability)
    MD2_dict[i]=MD2_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MD2_Identifiability.csv"),MD2_dict)

println()

println("MD3")
println(MD3_identifiability)
MD3_dict = Dict()
for i in keys(MD3_identifiability)
    MD3_dict[i]=MD3_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MD3_Identifiability.csv"),MD3_dict)

println()

println("MD4")
println(MD4_identifiability)

MD4_dict = Dict()
for i in keys(MD4_identifiability)
    MD4_dict[i]=MD4_identifiability[i]
end
CSV.write(string(DESTINATION_FOLDER,"MD4_Identifiability.csv"),MD4_dict)
println()

