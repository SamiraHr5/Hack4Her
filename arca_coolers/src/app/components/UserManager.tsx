"use client";

import { useEffect, useState } from "react";

interface CoolerData {
  cooler_id: string;
  temperature: number;
}

export default function UserManager() {
  const [coolers, setCoolers] = useState<CoolerData[]>([]);
  const [nombre, setNombre] = useState("");

  const fetchCoolers = async () => {
    try {
      const res = await fetch("/api/coolers");
      const data = await res.json();

      // Filtrar coolers con temperatura fuera del rango permitido
      const filtered = data.filter(
        (item: CoolerData) => item.temperature < 1.57 || item.temperature > 9.57
      );

      setCoolers(filtered);
    } catch (error) {
      console.error("Error al obtener datos de los coolers:", error);
    }
  };

  useEffect(() => {
    fetchCoolers();
  }, []);

  return (
    <div className="p-6 bg-white rounded shadow mt-12 max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold mb-6 text-[#7a3030] text-center">
        Coolers con Temperatura Anormal
      </h2>

      {coolers.length === 0 ? (
        <p className="text-gray-500 text-center">No hay alertas activas.</p>
      ) : (
        <ul className="divide-y divide-gray-200">
          {coolers.map((cooler, i) => (
            <li
              key={i}
              className="py-3 px-2 flex justify-between items-center hover:bg-gray-50 transition"
            >
              <span className="text-sm font-medium text-gray-700">
                Cooler ID: <span className="text-black">{cooler.cooler_id}</span>
              </span>
              <span
                className={`text-sm font-semibold ${
                  cooler.temperature < 1.57 ? "text-blue-600" : "text-red-600"
                }`}
              >
                {cooler.temperature.toFixed(2)} °C
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
