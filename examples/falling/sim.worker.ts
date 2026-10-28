// The physics of the example, off the main thread (see main.ts).
import { CableSim, hostSimWorker } from '@drschizzo/avbd-cables';

hostSimWorker((params) => new CableSim(params));
