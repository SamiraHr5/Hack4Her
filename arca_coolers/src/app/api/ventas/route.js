import { NextResponse } from "next/server";
import clientPromise from "../../components/lib/mongodb";

export async function GET() {
  try {
    const client = await clientPromise;
    const db = client.db("test"); // Cambia si tu base se llama diferente
    const collection = db.collection("ventas"); // Ajusta si tu colección tiene otro nombre

    const forecast = await collection
      .find({}, { projection: { cooler_id: 1, amount: 1, _id: 0 } })
      .toArray();

    return NextResponse.json(forecast);
  } catch (error) {
    console.error("Error al obtener ventas:", error);
    return NextResponse.json({ error: "Error al obtener forecast" }, { status: 500 });
  }
}