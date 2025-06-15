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
  LabelList,
} from "recharts";

/* ──────────── Tipos ──────────── */
interface Forecast {
  cooler_id: string;
  proba_mensual: number;
  amount?: number;
}

/* Colores corporativos */
const RISK_COLORS = ["#7f1d1d", "#b45309", "#166534"]; // más oscuros
const BAR_COLOR   = "#b91c1c";                         // rojo barras

export default function Page4() {
  const [forecastData, setForecastData] = useState<Forecast[]>([]);
  const [salesData,   setSalesData]   = useState<Forecast[]>([]);

  /* ---------- Cargar probas ---------- */
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/forecast");
        const json = await res.json();
        setForecastData(json);
      } catch (e) {
        console.error("Error al cargar predicciones:", e);
      }
    })();
  }, []);

  /* ---------- Cargar ventas ---------- */
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/ventas");
        const json = await res.json();
        setSalesData(json);
      } catch (e) {
        console.error("Error al cargar ventas:", e);
      }
    })();
  }, []);

  /* ---------- CSV ---------- */
  const downloadCSV = () => {
    const headers = ["cooler_id,proba_mensual"];
    const rows = forecastData.map(
      (d) => `${d.cooler_id},${(d.proba_mensual * 100).toFixed(2)}%`
    );
    const blob = new Blob([headers.concat(rows).join("\n")], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const a   = Object.assign(document.createElement("a"), {
      href: url,
      download: "prediccion_data.csv",
    });
    a.click();
    URL.revokeObjectURL(url);
  };

  /* ---------- Datos Pie ---------- */
  const pieData = [
    { name: "Riesgo Alto",  value: forecastData.filter((d) => d.proba_mensual >= 0.8).length },
    { name: "Riesgo Medio", value: forecastData.filter((d) => d.proba_mensual >= 0.5 && d.proba_mensual < 0.8).length },
    { name: "Riesgo Bajo",  value: forecastData.filter((d) => d.proba_mensual < 0.5).length },
  ];

  /* ---------- Datos Bar (Top-5) ---------- */
  const barData = [...salesData]
    .filter((d) => d.amount !== undefined)
    .sort((a, b) => (b.amount ?? 0) - (a.amount ?? 0))
    .slice(0, 5)
    .map((d) => ({
      name: d.cooler_id.slice(0, 6),      // abreviar
      amount: +(d.amount ?? 0).toFixed(2) // fuerza 2 dec.
    }));

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#f5f1eb] to-[#e8e3dd] text-neutral-800">
      {/* ---------- Header ---------- */}
      <header className="bg-white shadow-sm fixed top-0 w-full z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <Image src="/arca-logo.svg" alt="Arca Continental Logo" width={130} height={40} priority />
          <span className="absolute left-1/2 -translate-x-1/2 font-semibold text-lg text-[#7a3030]">
            Portal de Predicción
          </span>
        </div>
      </header>

      {/* ---------- Contenido ---------- */}
      <section className="pt-40 px-6 pb-20 max-w-7xl mx-auto">
        {/* botones */}
        <div className="flex justify-between items-center flex-wrap gap-4 mb-6">
          <Link href="/" className="bg-gradient-to-r from-[#9b1b1e] to-[#c0392b] text-white px-6 py-2 rounded-full shadow hover:opacity-90">
            ← Volver al inicio
          </Link>
          <button onClick={downloadCSV} className="bg-green-700 text-white px-6 py-2 rounded-full shadow hover:bg-green-800">
            Descargar CSV
          </button>
        </div>

        <h1 className="text-5xl font-bold text-[#7a3030] text-center drop-shadow mb-10">
          Predicción de Fallas y Ventas
        </h1>

        <div className="flex flex-col lg:flex-row gap-8">
          {/* ───── Column-Left: Gráficos ───── */}
          <div className="lg:w-1/2 flex flex-col gap-10">
            {/* Pie */}
            <div className="bg-white p-6 rounded shadow h-[420px]">
              <h2 className="text-lg font-semibold text-[#7a3030] text-center mb-4">
                Distribución de Riesgo de Falla
              </h2>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={120}
                    label={({ percent }) => `${(percent * 100).toFixed(1)}%`}
                  >
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={RISK_COLORS[i]} />
                    ))}
                  </Pie>
                  <Legend verticalAlign="bottom" height={36} />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Bar */}
            <div className="bg-white p-6 rounded shadow h-[420px]">
              <h2 className="text-lg font-semibold text-[#7a3030] text-center mb-4">
                Top 5 Pronóstico de Ventas por Cooler
              </h2>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barData}
                  margin={{ top: 20, right: 20, left: 0, bottom: 10 }}
                  barCategoryGap={30}
                >
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v: number) => `$${v.toLocaleString()}`} />
                  <Bar dataKey="amount" fill={BAR_COLOR} barSize={50} radius={[4, 4, 0, 0]}>
                    <LabelList dataKey="amount" position="top" formatter={(v: number) => `$${v.toLocaleString()}`} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Imagen de pérdida potencial */}
            <div className="bg-white p-6 rounded shadow">
              <h2 className="text-lg font-semibold text-[#7a3030] text-center mb-4">
                Pérdida potencial en mayo
              </h2>
              <div className="w-full flex justify-center">
                <img src="/perdida_potencial.png" alt="Gráfico de pérdida potencial" className="max-w-full h-auto rounded" />
              </div>
            </div>
          </div>

          {/* ───── Column-Right: Tabla ───── */}
          <div className="lg:w-1/2 bg-white rounded shadow overflow-x-auto">
            <table className="min-w-full text-sm text-left">
              <thead className="bg-[#7a3030] text-white uppercase tracking-wider text-xs">
                <tr>
                  <th className="px-6 py-3">Cooler ID</th>
                  <th className="px-6 py-3">Prob. Falla Mensual</th>
                </tr>
              </thead>
              <tbody>
                {forecastData.length ? (
                  forecastData.map((d, i) => (
                    <tr key={i} className="border-b hover:bg-gray-100">
                      <td className="px-6 py-4 break-all">{d.cooler_id}</td>
                      <td
                        className={`px-6 py-4 font-semibold ${
                          d.proba_mensual >= 0.8
                            ? "text-red-600"
                            : d.proba_mensual >= 0.5
                            ? "text-yellow-600"
                            : "text-green-600"
                        }`}
                      >
                        {(d.proba_mensual * 100).toFixed(2)}%
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
