// =====================================================================
// psov2_ninja.js  —  FAITHFUL port of psov2's working PSO Ninja model
// renderer (DashGL "Ninja Plugin", MIT/GPL by Kion).
//
// Source ported VERBATIM (parsing/math byte-for-byte) from:
//   _reference/psov2/public/lib/bitstream.js   -> BitStream
//   _reference/psov2/public/js/NinjaModel.js   -> NinjaModel (readBone,
//                                                  readChunk, weighted
//                                                  vertex parse, getModel,
//                                                  readAnim)
//   _reference/psov2/public/js/NinjaFile.js    -> the NJTL/NJCM/NMDM
//                                                  container entry (parse)
//
// The ONLY changes vs psov2 are THREE.js API adaptations for our bundled
// r160 (psov2 targeted r9x):
//   - geometry.addAttribute(...)          -> geometry.setAttribute(...)
//   - new MeshBasicMaterial({skinning})   -> drop `skinning` (r125+
//                                            SkinnedMesh auto-skins)
//   - vertexColors: THREE.VertexColors    -> vertexColors: true
//   - Object3D.applyMatrix(m)             -> Object3D.applyMatrix4(m)
//   - THREE.AnimationClip.parseAnimation  -> kept (still present in r160)
// The bone tree, chunk dispatch, NJD_*OFF type math, BAMS rotation
// (2*PI/0xFFFF), ZYX Euler order, 0x400 quaternion bone, the 0x28-0x30
// weighted-vertex accumulation, the 0x23/0x2a vertex-color heads, the
// strip-to-face winding, and the texId wiring are UNCHANGED.
//
// Textures: psov2 decoded PVR into THREE.Texture and indexed them by
// the per-material chunk texId. WE supply `opts.texList` (an array of
// THREE.Texture built from OUR already-decoded tiles) in EXACTLY that
// index contract — `getModel()` does `mat.map = texList[texId]`.
// =====================================================================

import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

// ---------------------------------------------------------------------
// BitStream  (verbatim port of _reference/psov2/public/lib/bitstream.js)
// ---------------------------------------------------------------------
function BitStream(name, data, bigEndian) {
  this.name = name;
  this.data = new DataView(data);
  this.littleEndian = bigEndian ? false : true;
  this.ofs = 0;
  this.view = this.data;
  this.store = {};
  this.length = data.byteLength;
}

BitStream.bitflag = function (uint, bitPosition) {
  return uint & (1 << bitPosition) ? true : false;
};

BitStream.bitmask = function (uint, bitsPositions) {
  let mask = 0;
  for (let i = 0; i < bitsPositions.length; i++) {
    mask |= uint & (1 << bitsPositions[i]);
  }
  return mask >> bitsPositions[0];
};

BitStream.prototype = {
  constructor: BitStream,

  setLittleEndian: function () {
    this.littleEndian = true;
  },

  setBigEndian: function () {
    this.littleEndian = false;
  },

  seekCur: function (whence) {
    this.ofs += whence;
  },

  seekSet: function (whence) {
    this.ofs = whence;
  },

  seekEnd: function (whence) {
    this.ofs = this.data.byteLength + whence;
  },

  tell: function () {
    return this.ofs;
  },

  tellf: function () {
    var str = this.ofs.toString(16);
    if (str.length % 2) {
      str = "0" + str;
    }
    return "0x" + str;
  },

  len: function () {
    return this.view.byteLength;
  },

  setOfs: function (len) {
    if (this.view !== this.data) {
      throw new Error("Cannot subdivide once range set");
    }
    let buffer = this.data.buffer;
    this.view = new DataView(buffer, this.ofs, len);
    this.ofs = 0;
  },

  storeOfs: function (key) {
    key = key.toString();
    this.store[key] = this.ofs;
  },

  restoreOfs: function (key) {
    key = key.toString();
    if (!this.store[key]) {
      throw new Error("Stored offset not in store");
    }
    this.ofs = this.store[key];
  },

  clearOfs: function () {
    let offset = this.view.byteOffset;
    let length = this.view.byteLength;
    this.ofs = offset + length;
    this.view = this.data;
    this.store = {};
  },

  readByte: function () {
    let byte = this.view.getUint8(this.ofs);
    this.ofs += 1;
    return byte;
  },

  readShort: function () {
    let short = this.view.getInt16(this.ofs, this.littleEndian);
    this.ofs += 2;
    return short;
  },

  readUShort: function () {
    let ushort = this.view.getUint16(this.ofs, this.littleEndian);
    this.ofs += 2;
    return ushort;
  },

  readInt: function () {
    let int = this.view.getInt32(this.ofs, this.littleEndian);
    this.ofs += 4;
    return int;
  },

  readUInt: function () {
    let uint = this.view.getUint32(this.ofs, this.littleEndian);
    this.ofs += 4;
    return uint;
  },

  readFloat: function () {
    let float = this.view.getFloat32(this.ofs, this.littleEndian);
    this.ofs += 4;
    return float;
  },

  readVec3: function () {
    return {
      x: this.readFloat(),
      y: this.readFloat(),
      z: this.readFloat(),
    };
  },

  readVec4: function () {
    return {
      x: this.readFloat(),
      y: this.readFloat(),
      z: this.readFloat(),
      w: this.readFloat(),
    };
  },

  readRot3: function () {
    return {
      x: this.readInt() * ((2 * Math.PI) / 0xffff),
      y: this.readInt() * ((2 * Math.PI) / 0xffff),
      z: this.readInt() * ((2 * Math.PI) / 0xffff),
    };
  },

  readShortRot3: function () {
    return {
      x: this.readUShort() * ((2 * Math.PI) / 0xffff),
      y: this.readUShort() * ((2 * Math.PI) / 0xffff),
      z: this.readUShort() * ((2 * Math.PI) / 0xffff),
    };
  },

  readColor: function () {
    return {
      b: this.readByte() / 255,
      g: this.readByte() / 255,
      r: this.readByte() / 255,
      a: this.readByte() / 255,
    };
  },

  readString: function (len) {
    var str = "";
    if (len) {
      for (let i = 0; i < len; i++) {
        let char = this.readByte();
        str += String.fromCharCode(char);
      }
    } else {
      while (this.ofs < this.view.byteLength) {
        let char = this.readByte();
        if (char === 0) {
          break;
        }
        str += String.fromCharCode(char);
      }
    }
    return str.replace(/\0/g, "");
  },

  copy: function (len) {
    let data = this.view.buffer.slice(this.ofs, this.ofs + len);
    this.ofs += len;
    return data;
  },

  seekNext: function () {
    if (this.ofs === this.view.byteLength) {
      return;
    }
    let byte;
    do {
      byte = this.view.getUint8(this.ofs);
      if (byte) {
        break;
      }
      this.ofs += 1;
    } while (this.ofs < this.view.byteLength);
  },

  toBlob: function () {
    return new Blob([this.view.buffer], { type: "application/octet-stream" });
  },
};

// ---------------------------------------------------------------------
// NinjaModel  (verbatim port of _reference/psov2/public/js/NinjaModel.js)
// Only THREE r160 API calls adapted — see header. Parsing math UNCHANGED.
// ---------------------------------------------------------------------
class NinjaModel {
  constructor(name, tex) {
    this.name = name;
    this.bones = [];
    this.texList = tex || [];
    this.matList = [];
    this.vertexStack = [];
    this.animList = [];

    this.v = [];

    this.vertices = [];
    this.matIndex = [];
    this.colors = [];
    this.vcolors = [];
    this.uv = [];
    this.skinWeight = [];
    this.skinIndex = [];
  }

  parse(bs, forceMerge) {
    let len;
    this.bs = bs;

    do {
      const magic = this.bs.readString(4);
      switch (magic) {
        case "NJTL":
          len = this.bs.readUInt();
          this.bs.seekCur(len);

          break;
        case "NJCM":
          if (this.bones.length && !forceMerge) {
            this.bs.seekCur(-4);
            return;
          } else if (forceMerge) {
            this.bones = [];
          }

          len = this.bs.readUInt();
          this.bs.setOfs(len);
          this.readBone();
          this.bs.clearOfs();

          break;
        case "NMDM":
          len = this.bs.readUInt();
          this.bs.setOfs(len);
          this.readAnim();
          this.bs.clearOfs();
          break;
      }
    } while (this.bs.tell() < this.bs.len() - 4);
  }

  rel(type, bs) {
    if (bs) {
      this.bs = bs;
    }

    switch (type) {
      case "NJCM":
        this.readBone();
        break;
      case "NMDM":
        this.readAnim();
        break;
    }
  }

  addAnimation(bs) {
    this.parse(bs);
  }

  getModel() {
    let geometry = new THREE.BufferGeometry();
    let vertices = new Float32Array(this.vertices);
    let colors = new Float32Array(this.colors);
    let vcolors = new Float32Array(this.vcolors);
    let uv = new Float32Array(this.uv);
    let skinWeight = new Float32Array(this.skinWeight);
    let skinIndex = new Uint16Array(this.skinIndex);

    // ADAPT (r160): addAttribute -> setAttribute (renamed in r123).
    geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(vcolors, 4));
    geometry.setAttribute("vcolor", new THREE.BufferAttribute(vcolors, 4));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geometry.setAttribute("skinWeight", new THREE.BufferAttribute(skinWeight, 4));
    geometry.setAttribute("skinIndex", new THREE.BufferAttribute(skinIndex, 4));

    geometry.computeVertexNormals();

    let stack = [];

    for (let i = 0; i < this.matIndex.length; i++) {
      let last = stack[stack.length - 1] || {};

      if (this.matIndex[i] === last.materialIndex) {
        last.count += 3;
        continue;
      }

      stack.push({
        start: i * 3,
        count: 3,
        materialIndex: this.matIndex[i],
      });
    }

    stack.forEach((m) => {
      geometry.addGroup(m.start, m.count, m.materialIndex);
    });

    let materials = [];
    for (let i = 0; i < this.matList.length; i++) {
      // ADAPT (r160): drop `skinning:true` (removed in r125; a
      // SkinnedMesh now auto-skins). Use `vertexColors: true`
      // (THREE.VertexColors constant removed in r125).
      let mat = new THREE.MeshBasicMaterial({
        vertexColors: true,
      });

      if (this.matList[i].blending) {
        mat.blending = 2;
      }

      if (this.matList[i].doubleSide) {
        mat.side = THREE.DoubleSide;
      }

      if (this.matList[i].texId !== -1 && this.texList[this.matList[i].texId]) {
        mat.map = this.texList[this.matList[i].texId];
        if (mat.map.transparent) {
          mat.transparent = true;
          mat.alphaTest = 0.05;
        }
      }

      materials.push(mat);
    }

    let mesh = new THREE.SkinnedMesh(geometry, materials);
    mesh.name = this.name;
    let armSkeleton = new THREE.Skeleton(this.bones);
    let rootBone = armSkeleton.bones[0];
    mesh.add(rootBone);
    mesh.bind(armSkeleton);
    mesh.geometry.animations = this.animList;
    return mesh;
  }

  readBone(parentBone) {
    //  Read bone structure

    const bone = {
      ofs: this.bs.tellf(),
      flag: this.bs.readUInt(),
    };

    const { flag } = bone;

    if (flag & 0x400) {
      bone.chunkOfs = this.bs.readUInt();
      bone.pos = this.bs.readVec3();
      bone.rot = this.bs.readVec3();
      bone.scl = this.bs.readVec3();
      bone.childOfs = this.bs.readUInt();
      bone.siblingOfs = this.bs.readUInt();
      bone.rot.w = this.bs.readFloat();
    } else {
      bone.chunkOfs = this.bs.readUInt();
      bone.pos = this.bs.readVec3();
      bone.rot = this.bs.readRot3();
      bone.scl = this.bs.readVec3();
      bone.childOfs = this.bs.readUInt();
      bone.siblingOfs = this.bs.readUInt();
    }

    // Create New Bone for Three.js

    let num = this.bones.length.toString();
    while (num.length < 3) {
      num = "0" + num;
    }

    this.bone = new THREE.Bone();
    this.bone.name = "bone_" + num;
    this.bones.push(this.bone);

    // Check flags for LightWave 3d Export

    let zxy = BitStream.bitflag(bone.flag, 5);
    if (zxy) {
      console.error("ZXY FLAG SET!!!!");
    }

    // Update bone local transform matrix

    if (!BitStream.bitflag(bone.flag, 2)) {
      this.bone.scale.x = bone.scl.x;
      this.bone.scale.y = bone.scl.y;
      this.bone.scale.z = bone.scl.z;
    }

    if (!BitStream.bitflag(bone.flag, 1)) {
      if (!bone.rot.w) {
        // ZYX Euler: Rz * Ry * Rx applied in X, Y, Z order (psov2 order).
        const xRotMatrix = new THREE.Matrix4();
        xRotMatrix.makeRotationX(bone.rot.x);
        this.bone.applyMatrix4(xRotMatrix); // ADAPT: applyMatrix -> applyMatrix4

        const yRotMatrix = new THREE.Matrix4();
        yRotMatrix.makeRotationY(bone.rot.y);
        this.bone.applyMatrix4(yRotMatrix);

        const zRotMatrix = new THREE.Matrix4();
        zRotMatrix.makeRotationZ(bone.rot.z);
        this.bone.applyMatrix4(zRotMatrix);
      } else {
        const { x, y, z, w } = bone.rot;
        const q = new THREE.Quaternion(x, y, z, w);
        const rotMatrix = new THREE.Matrix4();
        rotMatrix.makeRotationFromQuaternion(q);
        this.bone.applyMatrix4(rotMatrix); // ADAPT: applyMatrix -> applyMatrix4
      }
    }

    if (!BitStream.bitflag(bone.flag, 0)) {
      this.bone.position.x = bone.pos.x;
      this.bone.position.y = bone.pos.y;
      this.bone.position.z = bone.pos.z;
    }

    this.bone.updateMatrix();
    this.bone.updateMatrixWorld();

    // If parent Bone exists, add bone as child

    if (parentBone) {
      parentBone.add(this.bone);
      this.bone.updateMatrix();
      this.bone.updateMatrixWorld();
    }

    // If polygon exists for bone, seek and read

    if (bone.chunkOfs) {
      this.bs.seekSet(bone.chunkOfs);
      let vertexOfs = this.bs.readUInt();
      let stripOfs = this.bs.readUInt();

      if (vertexOfs) {
        this.bs.seekSet(vertexOfs);
        this.readChunk();
      }

      if (stripOfs) {
        this.bs.seekSet(stripOfs);
        this.readChunk();
      }
    }

    // If child bone exists, read and pass in current bone

    if (bone.childOfs) {
      this.bs.seekSet(bone.childOfs);
      this.readBone(this.bone);
    }

    // If sibling bone exists, read and pass in parent bone

    if (bone.siblingOfs) {
      this.bs.seekSet(bone.siblingOfs);
      this.readBone(parentBone);
    }
  }

  readChunk() {
    const NJD_NULLOFF = 0x00;
    const NJD_BITSOFF = 0x01;
    const NJD_TINYOFF = 0x08;
    const NJD_MATOFF = 0x10;
    const NJD_VERTOFF = 0x20;
    const NJD_VOLOFF = 0x38;
    const NJD_STRIPOFF = 0x40;
    const NJD_ENDOFF = 0xff;

    this.mat = {
      texId: -1,
      blending: false,
      doubleSide: false,
    };

    let chunk;

    this.color = {
      r: 1,
      g: 1,
      b: 1,
      a: 1,
    };

    do {
      chunk = {
        head: this.bs.readByte(),
        flag: this.bs.readByte(),
      };

      // Invalid Chunk

      if (chunk.head > NJD_STRIPOFF + 11) {
        continue;
      }

      // Strip Chunk

      if (chunk.head >= NJD_STRIPOFF) {
        this.readStripChunk(chunk);
        continue;
      }

      // Volume Chunk

      if (chunk.head >= NJD_VOLOFF) {
        throw new Error("Volume chunk detected");
      }

      // Vertex Chunk

      if (chunk.head >= NJD_VERTOFF) {
        this.readVertexChunk(chunk);
        continue;
      }

      // Material Chunk

      if (chunk.head >= NJD_MATOFF) {
        this.readMaterialChunk(chunk);
        continue;
      }

      // Tiny Chunk

      if (chunk.head >= NJD_TINYOFF) {
        this.readTinyChunk(chunk);
        continue;
      }

      // Bits Chunk

      if (chunk.head >= NJD_BITSOFF) {
        this.readBitsChunk(chunk);
        continue;
      }

      // End
    } while (chunk.head !== NJD_ENDOFF);

    if (this.mem_stack && this.mem_stack.length) {
      let ofs = this.mem_stack.pop();
      this.bs.seekSet(ofs);
      this.readChunk();
    }
  }

  readBitsChunk(chunk) {
    switch (chunk.head) {
      case 1:
        let dstAlpha = BitStream.bitmask(chunk.flag, [0, 1, 2]);
        let srcAlpha = BitStream.bitmask(chunk.flag, [3, 4, 5]);

        if (srcAlpha === 4 && dstAlpha === 1) {
          this.mat.blending = true;
        } else {
          this.mat.blending = false;
        }

        break;
      case 2:
        let mipmapDepth = chunk.flag & 0x0f;
        break;
      case 3:
        let specularCoef = chunk.flag & 0x1f;
        break;
      case 4:
        this.bs.storeOfs(chunk.flag);
        chunk.head = 0xff;
        break;
      case 5:
        this.mem_stack = this.mem_stack || [];
        this.mem_stack.push(this.bs.tell());
        this.bs.restoreOfs(chunk.flag);
        break;
    }
  }

  readTinyChunk(chunk) {
    let tinyChunk = this.bs.readUShort();
    chunk.textureId = tinyChunk & 0x1fff;

    this.mat.texId = chunk.textureId;

    let superSample = BitStream.bitflag(tinyChunk, 13);
    let filterMode = BitStream.bitmask(tinyChunk, [14, 15]);

    let clampU = BitStream.bitflag(chunk.head, 4);
    let clampV = BitStream.bitflag(chunk.head, 5);
    let flipU = BitStream.bitflag(chunk.head, 6);
    let flipV = BitStream.bitflag(chunk.head, 7);

    //this.flipV = flipV;
  }

  readMaterialChunk(chunk) {
    let r, g, b, a;

    chunk.length = this.bs.readUShort();

    // Alpha Blending Instructions

    let dstAlpha = BitStream.bitmask(chunk.flag, [0, 1, 2]);
    let srcAlpha = BitStream.bitmask(chunk.flag, [3, 4, 5]);

    if (srcAlpha === 4 && dstAlpha === 1) {
      this.mat.blending = true;
    } else {
      this.mat.blending = false;
    }

    let diffuse, specular, ambient, type;

    // Diffuse

    if (BitStream.bitflag(chunk.head, 0)) {
      this.color = {
        b: this.bs.readByte() / 255,
        g: this.bs.readByte() / 255,
        r: this.bs.readByte() / 255,
        a: this.bs.readByte() / 255,
      };
    }

    // Specular

    if (BitStream.bitflag(chunk.head, 1)) {
      //type = "phong";
      specular = this.bs.readColor();
      specular = null;
    }

    // Ambient

    if (BitStream.bitflag(chunk.head, 2)) {
      //type = type || "lambert";
      ambient = this.bs.readColor();
      ambient = null;
    }
  }

  readVertexChunk(chunk) {
    let r, g, b, a;

    chunk.length = this.bs.readUShort();

    // Read the index offset and the number of index

    let indexOfs = this.bs.readUShort();
    let nbIndex = this.bs.readUShort();

    // Read the vertex list

    for (let i = 0; i < nbIndex; i++) {
      let stackOfs = indexOfs + i;
      let vertex = new THREE.Vector3();

      // Read the position

      let pos = this.bs.readVec3();
      vertex.x = pos.x;
      vertex.y = pos.y;
      vertex.z = pos.z;
      vertex.applyMatrix4(this.bone.matrixWorld);

      // Read vertex normals

      if (chunk.head > 0x28 && chunk.head < 0x30) {
        let norm = this.bs.readVec3();
        let normal = new THREE.Vector3();
        normal.x = norm.x;
        normal.y = norm.y;
        normal.z = norm.z;
        //normal.applyMatrix3(this.bone.normalMatrix);
        vertex.normal = normal;
      }

      // Read vertex color

      if (chunk.head === 0x23 || chunk.head === 0x2a) {
        vertex.color = {
          b: this.bs.readByte() / 255,
          g: this.bs.readByte() / 255,
          r: this.bs.readByte() / 255,
          a: this.bs.readByte() / 255,
        };
        //throw new Error("Model has vertex color don't use this");
      }

      // Read Vertex weight

      let skinWeights = new THREE.Vector4(0, 0, 0, 0);
      let skinIndices = new THREE.Vector4(0, 0, 0, 0);

      if (chunk.head !== 0x2c) {
        skinIndices.x = this.bones.length - 1;
        skinWeights.x = 1.0;
      } else {
        // Read weight values

        let ofs = this.bs.readUShort();
        let weight = this.bs.readUShort();

        // Update current stack position

        stackOfs = indexOfs + ofs;

        // Set the vertex weights

        // If a previous vertex exists, get values

        if (this.vertexStack[stackOfs]) {
          let prev = this.vertexStack[stackOfs];
          let keys = ["x", "y", "z"];
          keys.forEach((axis) => {
            skinWeights[axis] = prev.skinWeight[axis];
            skinIndices[axis] = prev.skinIndice[axis];
          });
        }

        switch (chunk.flag) {
          case 0x80:
            skinIndices.x = this.bones.length - 1;
            skinWeights.x = weight / 255;
            break;
          case 0x81:
            skinIndices.y = this.bones.length - 1;
            skinWeights.y = weight / 255;
            break;
          case 0x82:
            skinIndices.z = this.bones.length - 1;
            skinWeights.z = weight / 255;
            break;
        }
      }

      // If the global index is set, continue

      // Push the vertex to the stack

      vertex.globalIndex = this.v.length;
      vertex.skinWeight = skinWeights;
      vertex.skinIndice = skinIndices;

      this.vertexStack[stackOfs] = vertex;
      this.v.push(vertex);
    }
  }

  getMaterialIndex() {
    for (let i = 0; i < this.matList.length; i++) {
      if (this.mat.texId !== this.matList[i].texId) {
        continue;
      }

      if (this.mat.blending !== this.matList[i].blending) {
        continue;
      }

      if (this.mat.doubleSide !== this.matList[i].doubleSide) {
        continue;
      }

      return i;
    }

    let mat = {};
    for (let key in this.mat) {
      mat[key] = this.mat[key];
    }

    let matId = this.matList.length;
    this.matList.push(mat);
    return matId;
  }

  readStripChunk(chunk) {
    chunk.length = this.bs.readUShort();

    // Read the number of strips and user offset

    let stripChunk = this.bs.readUShort();
    let nbStrips = stripChunk & 0x3fff;
    let userOffset = BitStream.bitmask(stripChunk, [14, 15]);

    this.mat.doubleSide = BitStream.bitflag(chunk.flag, 4);
    let index = this.getMaterialIndex();

    // Read the list of strips

    for (let i = 0; i < nbStrips; i++) {
      // Read the length and direction

      let strip_length = this.bs.readShort();
      let clockwise = strip_length < 0 ? true : false;
      let length = Math.abs(strip_length);
      let strip = new Array(length);

      // Read the strip

      for (let k = 0; k < strip.length; k++) {
        // Read stack position

        let stackOfs = this.bs.readUShort();

        strip[k] = {
          vertex: this.vertexStack[stackOfs],
        };

        // Read face uv values

        switch (chunk.head) {
          case 0x41:
            if (!this.flipV) {
              strip[k].uv = new THREE.Vector2(
                this.bs.readShort() / 255,
                1 - this.bs.readShort() / 255,
              );
            } else {
              strip[k].uv = new THREE.Vector2(
                this.bs.readShort() / 255,
                this.bs.readShort() / 255,
              );
            }

            break;
          case 0x42:
            if (!this.flipV) {
              strip[k].uv = new THREE.Vector2(
                this.bs.readShort() / 1023,
                1 - this.bs.readShort() / 1023,
              );
            } else {
              strip[k].uv = new THREE.Vector2(
                this.bs.readShort() / 1023,
                this.bs.readShort() / 1023,
              );
            }

            break;
          default:
            strip[k].uv = new THREE.Vector2();

            break;
        }

        // Seek passed user offset

        if (userOffset && k > 1) {
          this.bs.seekCur(userOffset * 2);
        }
      }

      // Convert strips into faces

      for (let k = 0; k < strip.length - 2; k++) {
        let a, b, c;
        let aPos, bPos, cPos;
        let aClr, bClr, cClr;
        let aUv, bUv, cUv;
        let aIdx, bIdx, cIdx;
        let aWgt, bWgt, cWgt;

        if ((clockwise && !(k % 2)) || (!clockwise && k % 2)) {
          a = strip[k + 0];
          b = strip[k + 2];
          c = strip[k + 1];
        } else {
          a = strip[k + 0];
          b = strip[k + 1];
          c = strip[k + 2];
        }

        this.matIndex.push(index);

        // Positions

        aPos = a.vertex;
        bPos = b.vertex;
        cPos = c.vertex;

        this.vertices.push(aPos.x, aPos.y, aPos.z);
        this.vertices.push(bPos.x, bPos.y, bPos.z);
        this.vertices.push(cPos.x, cPos.y, cPos.z);

        // Colors

        aClr = aPos.color || this.color;
        bClr = bPos.color || this.color;
        cClr = cPos.color || this.color;

        if (aClr.a < 0.3) {
          aClr.a = 0.3;
        }

        this.vcolors.push(aClr.r, aClr.g, aClr.b, aClr.a);
        this.vcolors.push(bClr.r, bClr.g, bClr.b, bClr.a);
        this.vcolors.push(cClr.r, cClr.g, cClr.b, cClr.a);

        // UV Values — V-flip (1 - v). psov2's UVs were authored for
        // flipY=true (THREE's TextureLoader default / DashGL's PVR), but
        // OUR tiles upload with flipY=false (PSOBB top-down V). Without
        // this flip the textures — most visibly the face — render upside
        // down. Flipping V here keeps the shared tile pipeline unchanged.

        this.uv.push(a.uv.x, 1.0 - a.uv.y);
        this.uv.push(b.uv.x, 1.0 - b.uv.y);
        this.uv.push(c.uv.x, 1.0 - c.uv.y);

        // Skin Index

        aIdx = aPos.skinIndice;
        bIdx = bPos.skinIndice;
        cIdx = cPos.skinIndice;

        this.skinIndex.push(aIdx.x, aIdx.y, aIdx.z, aIdx.w);
        this.skinIndex.push(bIdx.x, bIdx.y, bIdx.z, bIdx.w);
        this.skinIndex.push(cIdx.x, cIdx.y, cIdx.z, cIdx.w);

        // Skin Weight

        aWgt = aPos.skinWeight;
        bWgt = bPos.skinWeight;
        cWgt = cPos.skinWeight;

        this.skinWeight.push(aWgt.x, aWgt.y, aWgt.z, aWgt.w);
        this.skinWeight.push(bWgt.x, bWgt.y, bWgt.z, bWgt.w);
        this.skinWeight.push(cWgt.x, cWgt.y, cWgt.z, cWgt.w);
      }
    }
  }

  readAnim() {
    let motionOfs = this.bs.readUInt();
    let nbFrame = this.bs.readUInt();
    let motionType = this.bs.readUShort();
    let motionFlag = this.bs.readUShort();

    let nbElements = motionFlag & 0x0f;

    let motionList = new Array(this.bones.length);

    let motionTypes = {
      pos: BitStream.bitflag(motionType, 0),
      rot: BitStream.bitflag(motionType, 1),
      scl: BitStream.bitflag(motionType, 2),
      quat: motionType & 0x2000,
    };

    this.bs.seekSet(motionOfs);

    // Read offsets to animation list for each bone

    let firstOfs = this.bs.length;
    for (let i = 0; i < this.bones.length; i++) {
      let motionEntry = {
        bone: i,
        parent: i - 1,
        frames: [],
      };

      if (this.bs.tell() === firstOfs) {
        motionList[i] = motionEntry;
        continue;
        // break;
      }

      // Read the offset to each list

      for (let key in motionTypes) {
        if (!motionTypes[key]) {
          continue;
        }

        const ofs = this.bs.readUInt();
        if (ofs && ofs < firstOfs) {
          firstOfs = ofs;
        }

        motionEntry[key] = { ofs };
      }

      // Read the number of entries for each list

      for (let key in motionTypes) {
        if (!motionTypes[key]) {
          continue;
        }

        let num = this.bs.readUInt();

        if (num === 0) {
          delete motionEntry[key];
        } else {
          motionEntry[key].num = num;
        }
      }

      motionList[i] = motionEntry;
    }

    motionList.forEach((motion) => {
      // Read Position

      if (motion.pos) {
        this.bs.seekSet(motion.pos.ofs);

        for (let i = 0; i < motion.pos.num; i++) {
          let frameNo = this.bs.readUInt();
          let pos = this.bs.readVec3();

          if (!motion.frames[frameNo]) {
            motion.frames[frameNo] = {};
          }

          motion.frames[frameNo].pos = pos;
        }

        delete motion.pos;
      }

      // Read Rotation

      if (motion.rot) {
        this.bs.seekSet(motion.rot.ofs);

        for (let i = 0; i < motion.rot.num; i++) {
          let frameNo = this.bs.readUInt();
          let rot = this.bs.readRot3();

          if (!motion.frames[frameNo]) {
            motion.frames[frameNo] = {};
          }

          motion.frames[frameNo].rot = rot;
        }

        delete motion.rot;
      }

      if (motion.quat) {
        this.bs.seekSet(motion.quat.ofs);

        for (let i = 0; i < motion.quat.num; i++) {
          let frameNo = this.bs.readUInt();
          const w = this.bs.readFloat();
          const x = this.bs.readFloat();
          const y = this.bs.readFloat();
          const z = this.bs.readFloat();

          if (!motion.frames[frameNo]) {
            motion.frames[frameNo] = {};
          }

          motion.frames[frameNo].quat = [x, y, z, w];
        }

        delete motion.quat;
      }

      // Read Scale

      if (motion.scl) {
        this.bs.seekSet(motion.scl.ofs);

        for (let i = 0; i < motion.scl.num; i++) {
          let frameNo = this.bs.readUInt();
          let scl = this.bs.readVec3();

          if (!motion.frames[frameNo]) {
            motion.frames[frameNo] = {};
          }

          motion.frames[frameNo].scl = scl;
        }

        delete motion.scl;
      }
    });

    let animation = {
      name: this.bs.name.replace(".njm", ""),
      fps: 30,
      length: (nbFrame - 1) / 30,
      hierarchy: new Array(this.bones.length),
    };

    for (let i = 0; i < this.bones.length; i++) {
      let bone = this.bones[i];
      let motion = motionList[i];

      animation.hierarchy[i] = {
        parent: motion.parent,
        keys: [],
      };

      for (let k = 0; k < nbFrame; k++) {
        let frame = motion.frames[k];

        if (frame && frame.pos) {
          let pos = frame.pos;
          frame.pos = [pos.x, pos.y, pos.z];
        }

        if (frame && frame.rot) {
          let obj = new THREE.Bone();

          var xRotMatrix = new THREE.Matrix4();
          xRotMatrix.makeRotationX(frame.rot.x);
          obj.applyMatrix4(xRotMatrix); // ADAPT: applyMatrix -> applyMatrix4

          var yRotMatrix = new THREE.Matrix4();
          yRotMatrix.makeRotationY(frame.rot.y);
          obj.applyMatrix4(yRotMatrix);

          var zRotMatrix = new THREE.Matrix4();
          zRotMatrix.makeRotationZ(frame.rot.z);
          obj.applyMatrix4(zRotMatrix);

          let quat = new THREE.Quaternion();
          quat.setFromRotationMatrix(obj.matrix);
          frame.rot = quat.toArray();
        }

        if (frame && frame.quat) {
          frame.rot = frame.quat;
        }

        if (frame && frame.scl) {
          let scl = frame.scl;
          frame.scl = [scl.x, scl.y, scl.z];
        }

        if (k === 0 || k === nbFrame - 1) {
          frame = frame || {};

          if (!frame.pos) {
            frame.pos = bone.position.toArray();
          }

          if (!frame.rot) {
            frame.rot = bone.quaternion.toArray();
          }

          if (!frame.scl) {
            frame.scl = bone.scale.toArray();
          }
        }

        if (!frame) {
          continue;
        }

        frame.time = k / 30;

        animation.hierarchy[i].keys.push(frame);
      }
    }

    var clip = THREE.AnimationClip.parseAnimation(animation, this.bones);
    clip.optimize();
    this.animList.push(clip);
  }
}

// ---------------------------------------------------------------------
// Public entry point.
//
// Mirrors the psov2 orchestration (see AssetEnemies.js "Rappy"):
//   let modelLoader = new NinjaModel(name, tex);
//   modelLoader.parse(bml["..._base.nj"]);
//   modelLoader.addAnimation(bml["..._base.njm"]);  // 0..N
//   let mdl = modelLoader.getModel();
//
// opts:
//   name      : string  — mesh.name (default "ninja_model")
//   texList   : Array<THREE.Texture>  indexed by chunk texId (OUR tiles)
//   motions   : Array<ArrayBuffer>    raw .njm buffers (optional)
//
// Returns a THREE.SkinnedMesh (real Bone tree + Skeleton, bound), exactly
// as psov2's getModel() does. Animations land on mesh.geometry.animations.
// ---------------------------------------------------------------------
export function parseNinjaModel(arrayBuffer, opts) {
  opts = opts || {};
  const name = opts.name || "ninja_model";
  const texList = opts.texList || [];

  const bs = new BitStream(name, arrayBuffer);
  const modelLoader = new NinjaModel(name, texList);
  modelLoader.parse(bs);

  // Optional animations (.njm raw buffers), driven the same way psov2
  // does via addAnimation(BitStream).
  const motions = opts.motions || [];
  for (let i = 0; i < motions.length; i++) {
    try {
      const mbs = new BitStream(
        (opts.motionNames && opts.motionNames[i]) || `motion_${i}.njm`,
        motions[i],
      );
      modelLoader.addAnimation(mbs);
    } catch (e) {
      // A single bad motion must not sink the static bind pose.
      console.warn(`psov2_ninja: motion ${i} parse failed:`, e);
    }
  }

  const mesh = modelLoader.getModel();
  // Expose the loader for callers that want matList/texList introspection
  // (e.g. the texture/material panel) without re-parsing.
  mesh.userData.ninjaLoader = modelLoader;
  return mesh;
}

export { NinjaModel, BitStream };
