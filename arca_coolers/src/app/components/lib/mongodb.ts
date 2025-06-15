import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI;

if (!uri) {
  throw new Error("Falta la variable de entorno MONGODB_URI");
}

const options = {};

declare global {
  // Esto permite que Next.js reutilice el cliente durante hot reload en desarrollo
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

// Si ya existe una promesa previa, la reutilizamos (modo dev)
const clientPromise: Promise<MongoClient> =
  global._mongoClientPromise ??
  (global._mongoClientPromise = new MongoClient(uri, options).connect());

export default clientPromise;

