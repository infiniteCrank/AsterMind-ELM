// Presets.ts - Reusable configuration presets for ELM

import { ELMConfig } from '../core/ELMConfig';

interface ExtendedELMConfig extends ELMConfig {
    charSet?: string;
    useTokenizer?: boolean;
    tokenizerDelimiter?: RegExp;
}

export const EnglishCharPreset: ExtendedELMConfig = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 15,
    activation: 'relu',
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    useTokenizer: false,
    log: {
    }
};

export const EnglishTokenPreset: ExtendedELMConfig = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 20,
    activation: 'relu',
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    useTokenizer: true,
    tokenizerDelimiter: /[\s,.;!?()\[\]{}"']+/,
    log: {
    }
};

export const RussianCharPreset: ExtendedELMConfig = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 20,
    activation: 'relu',
    charSet: 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
    useTokenizer: false,
    log: {
    }
};

export const RussianTokenPreset: ExtendedELMConfig = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 20,
    activation: 'relu',
    charSet: 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
    useTokenizer: true,
    tokenizerDelimiter: /[\s,.;!?()\[\]{}"']+/,
    log: {
    }
};

export const EmojiCharPreset: ExtendedELMConfig = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 10,
    activation: 'relu',
    charSet: '😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚🙂🤗🤩',
    useTokenizer: false,
    log: {
    }
};

export const EmojiHybridPreset: ExtendedELMConfig = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 25,
    activation: 'relu',
    charSet: 'abcdefghijklmnopqrstuvwxyz😀😁😂🤣😃😄😅😆😉😊',
    useTokenizer: true,
    tokenizerDelimiter: /[\s,.;!?()\[\]{}"']+/,
    log: {
    }
};
