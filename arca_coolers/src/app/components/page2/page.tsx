"use client";

import Image from "next/image";
import Link from "next/link";
import React, { useEffect, useState } from "react";

export default function Page2() {
  const [alertas, setAlertas] = useState<any[]>([]);

  useEffect(() => {
    const fetchCoolers = async () => {
      const res = await fetch("/api/coolers");
      const data = await res.json();
      const alertaCoolers = data.filter(
        (item: any) => item.temperature < 1.57 || item.temperature > 9.57
      );
      setAlertas(alertaCoolers);
    };
    fetchCoolers();
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#f5f1eb] to-[#e8e3dd] text-neutral-800">
      {/* Encabezado */}
      <header className="bg-white shadow-sm fixed top-0 w-full z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          {/* Logo a la izquierda */}
          <div className="flex items-center gap-3">
            <Image
              src="/arca-logo.svg"
              alt="Arca Continental Logo"
              width={130}
              height={40}
              priority
            />
          </div>

          {/* Título centrado */}
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <span className="font-semibold text-lg text-[#7a3030]">
              Sistema de Predicción
            </span>
          </div>
        </div>
      </header>

      {/* Contenido */}
      <section className="pt-40 px-6 pb-12 max-w-7xl mx-auto text-center">
        {/* Título */}
        <h1 className="text-5xl font-bold text-[#7a3030] text-center drop-shadow">
          Soporte de Coolers
        </h1>
        <p className="max-w-2xl mx-auto text-lg text-neutral-700 mb-10">
          Sistema para creación y gestión de soporte a fallas
        </p>

        {/* Botón de regreso */}
        <div className="text-center mb-10">
          <Link
            href="/"
            className="inline-block bg-gradient-to-r from-[#9b1b1e] to-[#c0392b] text-white px-6 py-3 rounded-full shadow hover:opacity-90 transition"
          >
            ← Volver al inicio
          </Link>
        </div>

        {/* Alertas de temperatura */}
        {alertas.length > 0 && (
          <div className="mb-12">
            <h2 className="text-2xl font-semibold text-red-700 mb-4">Coolers con temperatura fuera de rango</h2>
            <ul className="space-y-4">
              {alertas.map((item, i) => (
                <li key={i} className="bg-white rounded-lg p-4 shadow text-left">
                  <p className="font-semibold text-[#7a3030]">Cooler ID:</p>
                  <p className="text-sm mb-2">{item.cooler_id}</p>
                  <p className="text-sm text-red-600">Temperatura: {item.temperature}°C</p>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Métricas generales y paneles */}
        {/* Aquí iría el resto del contenido que ya tenías */}
      </section>
    </main>
  );
}
