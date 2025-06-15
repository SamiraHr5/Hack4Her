"use client";

import Image from "next/image";
import Link from "next/link";
import React, { useEffect, useState } from "react";

interface Cooler {
  cooler_id: string;
  temperature: number;
}

interface Ticket {
  cooler_id: string;
  temperature: number;
  assigned_to?: string;
}

const TECHNICIANS = ["T-001 Juan", "T-002 Marta", "T-003 Luis", "T-004 Sofía"];

export default function Page2() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [draggedTechnician, setDraggedTechnician] = useState<string | null>(null);

  useEffect(() => {
    const fetchCoolers = async () => {
      const res = await fetch("/api/coolers");
      const data = await res.json();
      const alertaCoolers = data.filter(
        (item: Cooler) => item.temperature < 1.57 || item.temperature > 9.57
      );
      setTickets(alertaCoolers.map((item: Cooler) => ({ cooler_id: item.cooler_id, temperature: item.temperature })));
    };
    fetchCoolers();
  }, []);

  const handleDrop = (coolerId: string) => {
    if (!draggedTechnician) return;
    setTickets((prev) =>
      prev.map((t) =>
        t.cooler_id === coolerId ? { ...t, assigned_to: draggedTechnician } : t
      )
    );
    setDraggedTechnician(null);
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#f5f1eb] to-[#e8e3dd] text-neutral-800">
      {/* Encabezado */}
      <header className="bg-white shadow-sm fixed top-0 w-full z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <Image
            src="/arca-logo.svg"
            alt="Arca Continental Logo"
            width={130}
            height={40}
            priority
          />
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <span className="font-semibold text-lg text-[#7a3030]">
              Portal de Predicción
            </span>
          </div>
        </div>
      </header>

      {/* Contenido */}
      <section className="pt-40 px-6 pb-12 max-w-7xl mx-auto">
        <h1 className="text-5xl font-bold text-[#7a3030] text-center drop-shadow mb-10">
          Tickets de Soporte
        </h1>

        <div className="flex flex-col lg:flex-row gap-10">
          {/* Lista de tickets */}
          <div className="flex-1">
            <h2 className="text-xl font-semibold mb-4 text-red-700">
              Coolers con temperatura fuera de rango
            </h2>
            <ul className="space-y-4">
              {tickets.map((ticket, i) => (
                <li
                  key={i}
                  className="bg-white p-4 rounded shadow"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => handleDrop(ticket.cooler_id)}
                >
                  <p className="font-semibold text-[#7a3030]">Cooler ID: {ticket.cooler_id}</p>
                  <p className="text-sm text-red-600 mb-2">
                    Temperatura: {ticket.temperature}°C
                  </p>
                  <p className="text-sm">
                    Asignado a:{" "}
                    <span className="font-medium">
                      {ticket.assigned_to || "Sin asignar"}
                    </span>
                  </p>
                </li>
              ))}
            </ul>
          </div>

          {/* Columna de técnicos con botón fuera del cuadro */}
          <div className="w-full lg:w-1/3 flex flex-col items-center gap-4">
            {/* Botón fuera del recuadro */}
            <Link
              href="/"
              className="inline-block bg-gradient-to-r from-[#9b1b1e] to-[#c0392b] text-white px-6 py-2 rounded-full shadow hover:opacity-90 transition"
            >
              ← Volver al inicio
            </Link>

            {/* Técnicos */}
            <div className="w-full bg-white p-4 rounded shadow h-fit">
              <h2 className="text-xl font-semibold text-[#7a3030] mb-4 text-center">
                Técnicos disponibles
              </h2>
              <ul className="space-y-3">
                {TECHNICIANS.map((tech, i) => (
                  <li
                    key={i}
                    draggable
                    onDragStart={() => setDraggedTechnician(tech)}
                    className="cursor-move px-4 py-2 bg-red-100 text-red-800 font-medium rounded hover:bg-red-200 transition"
                  >
                    {tech}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
