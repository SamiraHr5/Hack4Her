"use client";

import Image from "next/image";
import Link from "next/link";
import TableauDashboard from "./components/Tableau";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#e8e3dd] text-neutral-800">
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
              Portal de Predicción
            </span>
          </div>
        </div>
      </header>

      {/* Contenido principal */}
      <section className="pt-40 pb-12 px-6 text-center">
        <h1 className="text-5xl font-bold text-[#7a3030] mb-6 drop-shadow">
          Historial de fallas en coolers
        </h1>
        <p className="max-w-2xl mx-auto text-lg text-neutral-700 mb-10">
          Anticipar fallas en sistemas de refrigeración mediante visualización de datos
        </p>

        {/* Botones de navegación */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-10">
          <Link
            href="/components/page2"
            className="inline-block bg-gradient-to-r from-[#9b1b1e] to-[#c0392b] text-white px-6 py-3 rounded-full shadow hover:opacity-90 transition"
          >
            Ir a sistema de tickets →
          </Link>

          <Link
            href="/components/page4"
            className="inline-block bg-gradient-to-r from-[#9b1b1e] to-[#c0392b] text-white px-6 py-3 rounded-full shadow hover:opacity-90 transition"
          >
            Predicción →
          </Link>

        </div>

        {/* Dashboard Tableau */}
        <div className="mt-10">
          <TableauDashboard />
        </div>
      </section>
    </main>
  );
}
