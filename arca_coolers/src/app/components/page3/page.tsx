import TableauDashboard from "../Tableau";
import Link from "next/link";
import Image from "next/image";

export default function Page3() {
  return (
    <main className="min-h-screen bg-[#e8e3dd] text-neutral-800">
      {/* Encabezado estilo Arca */}
      <header className="bg-white shadow-sm fixed top-0 w-full z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <Image
              src="/arca-logo.svg"
              alt="Arca Continental"
              width={130}
              height={40}
              priority
            />
            <span className="font-semibold text-lg text-[#7a3030]">
              Análisis de Corrupción
            </span>
          </div>

          <Link
            href="/"
            className="bg-gradient-to-r from-[#9b1b1e] to-[#c0392b] text-white font-medium px-4 py-2 rounded-full shadow hover:opacity-90 transition"
          >
            ← Volver al Inicio
          </Link>
        </div>
      </header>

      {/* Contenido principal */}
      <section className="pt-36 pb-20 px-4">
        <div className="max-w-6xl mx-auto bg-white rounded-3xl shadow-xl p-8">
          <h1 className="text-4xl font-bold mb-6 text-[#7a3030] text-center">
            Nuevo León Bajo la Lupa
          </h1>
          <p className="text-center text-lg text-neutral-600 max-w-3xl mx-auto mb-10">
            Observa y analiza visualmente los reportes de corrupción en tránsito mediante el tablero interactivo.
          </p>

          {/* Dashboard */}
          <TableauDashboard />
        </div>
      </section>
    </main>
  );
}
