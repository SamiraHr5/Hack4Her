import { NextResponse } from "next/server";
import clientPromise from "../../components/lib/mongodb";

export async function GET() {
  try {
    const client = await clientPromise;
    const db = client.db("test"); // Cambia si tu base se llama diferente
    const collection = db.collection("prediccion"); // Ajusta si tu colección tiene otro nombre

    const forecast = await collection
      .find({}, { projection: { cooler_id: 1, proba_mensual: 1, _id: 0 } })
      .toArray();

    return NextResponse.json(forecast);
  } catch (error) {
    console.error("Error al obtener forecast:", error);
    return NextResponse.json({ error: "Error al obtener forecast" }, { status: 500 });
  }
}
