"use client";

import Image from "next/image";
import Link from "next/link";
import React, { useEffect, useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface Forecast {
  cooler_id: string;
  proba_mensual: number;
  amount?: number;
}

const RISK_COLORS = ["#dc2626", "#facc15", "#16a34a"];

export default function Page4() {
  const [forecastData, setForecastData] = useState<Forecast[]>([]);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        const res = await fetch("/api/forecast");
        const data = await res.json();
        setForecastData(data);
      } catch (err) {
        console.error("Error al cargar predicciones:", err);
      }
    };

    fetchForecast();
  }, []);

  const downloadCSV = () => {
    const headers = ["cooler_id,proba_mensual"];
    const rows = forecastData.map(
      (item) =>
        `${item.cooler_id},${(item.proba_mensual * 100).toFixed(2)}%`
    );
    const csvContent = [...headers, ...rows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "forecast_data.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  // Pie chart risk data
  const pieData = [
    {
      name: "Riesgo Alto",
      value: forecastData.filter((d) => d.proba_mensual >= 0.8).length,
    },
    {
      name: "Riesgo Medio",
      value: forecastData.filter(
        (d) => d.proba_mensual >= 0.5 && d.proba_mensual < 0.8
      ).length,
    },
    {
      name: "Riesgo Bajo",
      value: forecastData.filter((d) => d.proba_mensual < 0.5).length,
    },
  ];

  // Bar chart data: top 5 by amount
  const topSales = [...forecastData]
    .filter((d) => d.amount !== undefined)
    .sort((a, b) => (b.amount ?? 0) - (a.amount ?? 0))
    .slice(0, 5)
    .map((item) => ({
      name: item.cooler_id.slice(0, 6),
      amount: item.amount ?? 0,
    }));

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#f5f1eb] to-[#e8e3dd] text-neutral-800">
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

      <section className="pt-40 px-6 pb-20 max-w-7xl mx-auto">
        <div className="flex justify-between items-center flex-wrap gap-4 mb-6">
          <Link
            href="/"
            className="bg-gradient-to-r from-[#9b1b1e] to-[#c0392b] text-white px-6 py-2 rounded-full shadow hover:opacity-90 transition"
          >
            ← Volver al inicio
          </Link>
          <button
            onClick={downloadCSV}
            className="bg-green-700 text-white px-6 py-2 rounded-full shadow hover:bg-green-800 transition"
          >
            Descargar CSV
          </button>
        </div>

        <h1 className="text-5xl font-bold text-[#7a3030] mb-6 text-center drop-shadow">
          Predicción de Fallas y Ventas
        </h1>

        <div className="flex flex-col lg:flex-row gap-8 mt-10">
          {/* Gráficos */}
          <div className="lg:w-1/2 flex flex-col gap-10">
            <div className="bg-white p-4 rounded shadow h-[400px]">
              <h2 className="text-lg font-semibold mb-4 text-[#7a3030] text-center">
                Distribución de Riesgo de Falla
              </h2>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  >
                    {pieData.map((_, index) => (
                      <Cell key={index} fill={RISK_COLORS[index]} />
                    ))}
                  </Pie>
                  <Legend />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-4 rounded shadow h-[400px]">
              <h2 className="text-lg font-semibold mb-4 text-[#7a3030] text-center">
                Top 5 Pronóstico de Ventas por Cooler
              </h2>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topSales}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="amount" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Tabla */}
          <div className="lg:w-1/2 bg-white rounded shadow overflow-x-auto">
            <table className="min-w-full text-sm text-left">
              <thead className="bg-[#7a3030] text-white uppercase tracking-wider text-xs">
                <tr>
                  <th className="px-6 py-3">Cooler ID</th>
                  <th className="px-6 py-3">Prob. Falla Mensual</th>
                </tr>
              </thead>
              <tbody>
                {forecastData.length > 0 ? (
                  forecastData.map((item, index) => (
                    <tr
                      key={index}
                      className="border-b hover:bg-gray-100 transition"
                    >
                      <td className="px-6 py-4 break-all">{item.cooler_id}</td>
                      <td
                        className={`px-6 py-4 font-semibold ${
                          item.proba_mensual >= 0.8
                            ? "text-red-600"
                            : item.proba_mensual >= 0.5
                            ? "text-yellow-600"
                            : "text-green-600"
                        }`}
                      >
                        {(item.proba_mensual * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={2} className="px-6 py-6 text-center text-gray-500">
                      No hay datos disponibles.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
  );
}
